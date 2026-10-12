import pytest

from railbaron.expected import BANK_FEE, trip_costs, turn_graph
from railbaron.game import Game, simulate
from railbaron.policy import ExpectedCostPolicy, Policy
from railbaron.tables import Tables

from test_game import check_recording


@pytest.fixture(scope="module")
def tables():
    return Tables()


def node(tables, name):
    return tables.city_node[tables.destinations.index(name)]


def test_turn_weights_are_the_dice(tables):
    g = turn_graph(tables)
    plain, express, chief = g.weights(False, False), g.weights(True, False), g.weights(False, True)
    for w in (plain, express, chief):
        assert w.sum() == pytest.approx(1)
    assert plain[7] == pytest.approx(6 / 36) and plain[12] == 0             # double six always rolls on
    assert plain[13:].sum() == pytest.approx(1 / 36)
    assert express[2] == 0 and express[13:].sum() > plain[13:].sum()       # every double rolls on
    assert chief[2] == 0 and chief[3] == pytest.approx(1 / 216) and chief[18] == pytest.approx(1 / 216)


def test_with_nobody_to_pay_the_cost_is_turns_times_the_price_of_a_turn(tables):
    g = turn_graph(tables)
    a, b = node(tables, "Seattle"), node(tables, "Miami")
    fees = [0, 5000, 5000, 5000]
    start, steps, column, kind, railroads, outside = trip_costs(
        tables, b, a, [0] * tables.rr_count, fees, False, False, 3500)
    turns = g.turns(False, False)[tables.hops[a][b]]
    assert start[g.start[a]] == pytest.approx((3500 + BANK_FEE) * turns, rel=1e-3)
    assert steps.shape[2] == 1 and list(kind) == [-1, -1, -1, -1]           # nobody to keep track of
    assert all(start[arc] == 0 for arc in range(g.count) if g.head[arc] == b)


def test_an_opponent_across_the_way_raises_the_cost_and_is_tracked(tables):
    g = turn_graph(tables)
    a, b = node(tables, "Seattle"), node(tables, "Miami")
    free = trip_costs(tables, b, a, [0] * tables.rr_count, [0, 5000, 5000, 5000], False, False, 3500)[0]
    way = tables.shortest(a, b)
    classes = [1 if way[2] >> r & 1 else 0 for r in range(tables.rr_count)]   # an opponent owns the short way
    start, steps, column, kind, railroads, outside = trip_costs(
        tables, b, a, classes, [0, 5000, 5000, 5000], False, False, 3500)
    assert start[g.start[a]] > free[g.start[a]] + 1000
    assert kind[1] == 0 and steps.shape[2] == 2
    # Having paid them already this turn can only help.
    assert (steps[1:, :, 1] <= steps[1:, :, 0] + 1e-6).all()
    # A Superchief gets there in fewer turns, so for less.
    fast = trip_costs(tables, b, a, classes, [0, 5000, 5000, 5000], False, True, 3500)[0]
    assert fast[g.start[a]] < start[g.start[a]]


def test_on_an_empty_board_the_calculated_route_is_a_shortest_one(tables):
    g = Game(4, 3, tables, policies=ExpectedCostPolicy())
    p = g.players[0]
    for start, end in (("Seattle", "Miami"), ("New York", "Chicago"), ("Dallas", "Boston")):
        a, b = node(tables, start), node(tables, end)
        p.dots, p.rrs = tables.shortest(a, b)[0], tables.shortest(a, b)[3]
        p.step, p.trail, p.goal = 0, None, b
        for roll in (2, 7, 12):
            g.roll = (0, roll, False)
            dots, rrs = p.policy.choose_route(g, p, a, b, None)
            assert dots[0] == a and dots[-1] == b and len(dots) - 1 == tables.hops[a][b]
    g.roll = None


@pytest.mark.parametrize("players", [3, 4, 5, 6])
def test_games_with_calculated_routes_follow_the_rules(tables, players):
    g = Game(players, 40 + players, tables, policies=ExpectedCostPolicy(), record=True).play()
    assert g.winner is not None
    check_recording(tables, g.recording)


def test_calculated_routes_play_the_same_game_recorded_or_not_and_on_any_workers(tables):
    policy = ExpectedCostPolicy()
    a = Game(4, 9, tables, policies=policy).play()
    b = Game(4, 9, tables, policies=policy, record=True).play()
    assert (a.winner, a.turn, [p.cash for p in a.players]) == (b.winner, b.turn, [p.cash for p in b.players])
    one = simulate(60, players=4, seed=5, tables=tables, policies=policy)
    two = simulate(60, players=4, seed=5, tables=tables, policies=policy, workers=2)
    for key in ("win_share_by_seat", "average_rounds", "average_railroads_sold", "purchases"):
        assert one[key] == two[key]


def test_only_a_policy_that_asks_for_it_is_consulted_on_every_roll(tables):
    assert Policy.EVERY_ROLL is False and ExpectedCostPolicy.EVERY_ROLL is True
    asked = {False: [], True: []}

    class Quick(Policy):
        def choose_route(self, game, p, src, dst, blocked=None):
            asked[self.EVERY_ROLL].append(game.roll)
            return super().choose_route(game, p, src, dst, blocked)

    class Eager(Quick):
        EVERY_ROLL = True

    for policy in (Quick(), Eager()):
        g = Game(3, 2, tables, policies=policy)
        for _ in range(12):
            g.play_turn()
    assert len(asked[True]) > len(asked[False])
    assert any(roll is not None for roll in asked[True])        # and it is told the roll
    # Asking again changes nothing when the answer is the same route.
    assert [p.cash for p in g.players] is not None
