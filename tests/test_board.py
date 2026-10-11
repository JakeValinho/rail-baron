import networkx as nx
import pytest

from railbaron import Board, FeeModel


@pytest.fixture(scope="module")
def board():
    return Board.load()


def test_map_loads_whole_and_connected(board):
    assert len(board.railroads) == 28
    assert len(board.cities) == 65
    assert board.graph.number_of_edges() == 748
    assert nx.is_connected(board.simple)


def test_destination_probabilities_sum_to_one(board):
    assert sum(board.destination_probability(c) for c in board.cities) == pytest.approx(1.0)


def test_small_railroads_serve_the_right_cities(board):
    assert board.cities_served(["RF+P"]) == {"Richmond", "Washington DC"}
    assert board.cities_served(["NYNH+H"]) == {"Boston", "New York"}
    assert board.cities_served(["B+M"]) == {"Albany", "Boston", "Portland ME"}


def test_distance_is_symmetric_and_accepts_names_or_ids(board):
    assert board.distance("New York", "Chicago") == 11
    assert board.distance("Chicago", "New York") == 11
    assert board.distance(board.node("Chicago"), "chicago") == 0


def test_shortest_route_is_a_real_path(board):
    route = board.shortest_route("Seattle", "Miami")
    assert route.distance == board.distance("Seattle", "Miami")
    assert board.label(route.start) == "Seattle" and board.label(route.end) == "Miami"
    for a, b, rr in zip(route.nodes, route.nodes[1:], route.railroads):
        assert rr in board.railroads_between(a, b)


def test_cheapest_route_avoids_opponents_and_prefers_own_track(board):
    ownership = {"NYC": "B", "PA": "B", "B+O": "A"}
    route = board.cheapest_route("New York", "Chicago", ownership, "A")
    assert "NYC" not in route.railroads
    assert "B+O" in route.railroads
    direct = board.shortest_route("New York", "Chicago")
    assert board.estimate_fee(route, ownership, "A") < board.estimate_fee(direct, ownership, "A")


def test_high_step_cost_gives_the_shortest_route(board):
    ownership = {"NYC": "B"}
    route = board.cheapest_route("New York", "Chicago", ownership, "A", step_cost=1e6)
    assert route.distance == 11


def test_turn_fee_charges_each_owner_once(board):
    ownership = {"NYC": "A", "PA": "B", "B+O": "B", "C+O": "C"}
    fee = lambda rrs: board.turn_fee(rrs, ownership, "A")
    assert fee(["NYC"]) == 0
    assert fee(["NYC", "ACL", "SAL"]) == 1000          # two bank lines, one fee
    assert fee(["PA", "B+O"]) == 5000                  # one opponent, one fee
    assert fee(["PA", "C+O", "ACL"]) == 11000
    assert board.turn_fee(["PA"], ownership, "A", FeeModel(opponent=10000)) == 10000


@pytest.mark.parametrize("start", ["Chicago", "Miami", "Kansas City", "NW_20"])
@pytest.mark.parametrize("roll", [1, 4, 7])
def test_reachable_matches_full_enumeration(board, start, roll):
    moves = board.legal_moves(start, roll)
    ends = board.reachable(start, roll)
    assert {m.end for m in moves} == set(ends)
    for move in moves:
        assert move.distance == roll
        assert len(set(move.tracks)) == roll  # no track ridden twice


def test_move_stops_at_destination_and_respects_used_track(board):
    first = board.shortest_route("New York", "Albany")
    assert first.distance == 2
    ends = board.reachable("New York", 6, destination="Albany")
    assert board.node("Albany") in ends
    assert ends[board.node("Albany")].distance <= 6
    stops = {m.distance for m in board.legal_moves("New York", 6, destination="Albany")
             if m.end == board.node("Albany")}
    assert 2 in stops  # arriving early ends the move

    # Having ridden NYC into Albany, the same NYC track cannot be ridden back.
    back = board.legal_moves("Albany", 1, used_tracks=first.tracks)
    assert all(m.tracks[0] not in first.tracks for m in back)


def test_traffic_adds_up_to_average_trip_length(board):
    prob = {c.node: board.destination_probability(c.name) for c in board.cities.values()}
    expected = sum(
        ps * pt / (1 - ps) * board.distance(s, t)
        for s, ps in prob.items()
        for t, pt in prob.items()
        if s != t
    )
    assert sum(board.traffic.values()) == pytest.approx(expected)


def test_railroad_stats_and_purchase_gain(board):
    rows = board.railroad_stats()
    assert len(rows) == 28
    assert sum(r["traffic_share"] for r in rows) == pytest.approx(1.0)
    assert sum(r["track"] for r in rows) == 748
    # Buying a line whose cities you already reach adds no coverage.
    assert board.purchase_gain(["PA"], "RF+P") == pytest.approx(
        board.destination_probability("Richmond")
    )
    assert board.purchase_gain(["PA", "NYC"], "PA") == 0
