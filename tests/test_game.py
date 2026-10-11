import random

import pytest

from railbaron.game import (EXPRESS_PRICE, FEE, FEE_ALL_SOLD, MOVE, ROVER_PRIZE, STARTING_CASH,
                            SUPERCHIEF_PRICE, TARGET_CASH, Game, record_games, simulate)
from railbaron.policy import BANK, BANK_FEE, EXPRESS, SUPERCHIEF, Policy
from railbaron.tables import Tables, search


@pytest.fixture(scope="module")
def tables():
    return Tables()


def scripted(values):
    """A stand-in for the game's dice: hands out `values`, then real random numbers."""
    queue, rng = list(values), random.Random(0)
    return lambda: queue.pop(0) if queue else rng.random()


def put_on_trip(g, p, start, end, plan=False):
    """Stand player p in city `start` with a trip to `end` along the shortest route."""
    d, t = g.dest, g.tables
    p.city = p.trip_from = d.index(start)
    p.destination = d.index(end)
    p.declared = p.at_home = False
    p.trail = p.est = None
    if plan:
        g._plan(p, t.city_node[p.city], t.city_node[p.destination], None, None)
    else:
        p.dots, _, _, p.rrs = t.shortest(t.city_node[p.city], t.city_node[p.destination])
        p.step, p.goal = 0, p.dots[-1]


def ride(g, p, count):
    g.bank_due, g.due = 0, {}
    g._move(p, count, None, [count])
    g._settle(p, None)


# ------------------------------------------------------------------ setup


def test_player_count_is_limited(tables):
    for n in (3, 4, 5, 6):
        assert len(Game(n, 1, tables).players) == n
    for n in (2, 7):
        with pytest.raises(ValueError):
            Game(n, 1, tables)


def test_everyone_starts_with_cash_and_a_different_home(tables):
    for seed in range(200):
        g = Game(6, seed, tables)
        assert all(p.cash == STARTING_CASH for p in g.players)
        homes = [tables.city_place[p.home] for p in g.players]
        assert len(set(homes)) == 6


def test_move_dice_are_two_fair_dice():
    assert len(MOVE) == 36 and min(MOVE) == 2 and max(MOVE) == 12
    assert MOVE.count(7) == 6 and MOVE.count(2) == 1


# ------------------------------------------------------------------- fees


def test_bank_and_own_track_cost_1000_a_turn(tables):
    g = Game(3, 1, tables)
    p = g.players[0]
    nyc = tables.rr_ids.index("NYC")
    put_on_trip(g, p, "New York", "Chicago")
    assert set(p.rrs) == {nyc}
    ride(g, p, 5)
    assert p.cash == STARTING_CASH - BANK_FEE          # the bank's line
    g._transfer(nyc, 0)
    ride(g, p, 2)
    assert p.cash == STARTING_CASH - 2 * BANK_FEE      # the player's own line still costs $1,000


def test_opponent_track_costs_5000_once_per_owner_per_turn(tables):
    g = Game(3, 1, tables)
    p, owner = g.players[0], g.players[1]
    nyc = tables.rr_ids.index("NYC")
    g._transfer(nyc, 1)
    put_on_trip(g, p, "New York", "Chicago")
    ride(g, p, 7)
    assert p.cash == STARTING_CASH - FEE and owner.cash == STARTING_CASH + FEE
    ride(g, p, 2)                                       # a new turn: charged again
    assert p.cash == STARTING_CASH - 2 * FEE

    # Two lines of one owner and a bank line in one turn: one fee each.
    pa = tables.rr_ids.index("PA")
    g._transfer(pa, 1)
    p.cash = STARTING_CASH
    bo = tables.rr_ids.index("B+O")
    p.dots, p.rrs, p.step = (0, 1, 2, 3, 4), (nyc, pa, bo, bo), 0
    p.destination = g.dest.index("Miami")
    ride(g, p, 2)
    assert p.cash == STARTING_CASH - FEE
    p.step = 0
    ride(g, p, 3)
    assert p.cash == STARTING_CASH - FEE - FEE - BANK_FEE


def test_fee_doubles_once_every_railroad_is_sold_and_stays_doubled(tables):
    g = Game(3, 1, tables)
    for r in range(tables.rr_count - 1):
        g._transfer(r, 1 + r % 2)
    assert g.rate == FEE
    g._transfer(tables.rr_count - 1, 1)
    assert g.rate == FEE_ALL_SOLD
    g._transfer(0, BANK)                                # sold back: the fee does not come down
    assert g.rate == FEE_ALL_SOLD


def test_established_player_keeps_the_old_fee_until_leaving_the_line(tables):
    g = Game(3, 1, tables)
    p, buyer = g.players[0], g.players[1]
    nyc, pa = tables.rr_ids.index("NYC"), tables.rr_ids.index("PA")
    put_on_trip(g, p, "New York", "Chicago")
    g._transfer(nyc, 1)                                 # bought while p stands on it
    assert p.est == {nyc: BANK_FEE}
    ride(g, p, 4)
    assert p.cash == STARTING_CASH - BANK_FEE           # still $1,000 ...
    assert buyer.cash == STARTING_CASH + BANK_FEE       # ... but paid to the new owner
    # Leave the line and come back in one turn: the full fee applies.
    p.rrs = p.rrs[:p.step] + (pa, nyc) + p.rrs[p.step + 2:]
    ride(g, p, 2)
    assert p.est is None
    assert p.cash == STARTING_CASH - BANK_FEE - BANK_FEE - FEE

    # A player elsewhere on the map is not established.
    g2 = Game(3, 1, tables)
    q = g2.players[0]
    put_on_trip(g2, q, "Los Angeles", "Seattle")
    g2._transfer(nyc, 1)
    assert q.est is None


def test_fee_rise_to_10000_also_leaves_players_established(tables):
    g = Game(3, 1, tables)
    p = g.players[0]
    nyc = tables.rr_ids.index("NYC")
    put_on_trip(g, p, "New York", "Chicago")
    for r in range(tables.rr_count):
        if r != 3:
            g._transfer(r, 1 + r % 2)
    p.est = None
    g._transfer(3, 2)                                   # the last line: fees double
    assert g.rate == FEE_ALL_SOLD
    assert p.est[nyc] == FEE
    ride(g, p, 3)
    assert p.cash == STARTING_CASH - FEE


# ------------------------------------------------------- selling and auctions


class Bids(Policy):
    def __init__(self, limit):
        self.limit = limit

    def max_bid(self, game, p, r):
        return self.limit


def test_auction_goes_round_the_table_in_500s_from_half_price(tables):
    acl = tables.rr_ids.index("ACL")                    # $12,000, so bidding opens at $6,000
    g = Game(3, 1, tables, policies=[Policy(), Bids(9000), Bids(20000)])
    g._transfer(acl, 0)
    buyer, price, bids = g._auction(g.players[0], acl)
    assert (buyer, price) == (2, 9500)
    assert bids[0] == [1, 6000] and bids[1] == [2, 6500]
    assert all(b[1] - a[1] == 500 for a, b in zip(bids, bids[1:]))

    g = Game(3, 1, tables, policies=[Policy(), Bids(9000), Bids(0)])
    g._transfer(acl, 0)
    assert g._auction(g.players[0], acl)[:2] == (1, 6000)

    g = Game(3, 1, tables, policies=[Policy(), Bids(5500), Bids(0)])
    g._transfer(acl, 0)
    assert g._auction(g.players[0], acl) == (BANK, 6000, [])    # nobody bids: the bank pays half

    g = Game(3, 1, tables, policies=[Policy(), Bids(50000), Bids(0)])
    g._transfer(acl, 0)
    g.players[1].cash = 7200                            # cannot bid more than they hold
    assert g._auction(g.players[0], acl)[:2] == (1, 6000)


def test_player_who_cannot_pay_sells_and_one_with_nothing_is_out(tables):
    acl = tables.rr_ids.index("ACL")
    g = Game(3, 1, tables, policies=[Policy(), Bids(0), Bids(0)])
    p = g.players[0]
    g._transfer(acl, 0)
    p.cash = 500
    g.bank_due, g.due = BANK_FEE, {1: FEE}
    g._settle(p, None)
    assert p.lines == [] and g.owner[acl] == BANK and not p.out
    assert p.cash == 500 + 6000 - BANK_FEE - FEE
    assert g.players[1].cash == STARTING_CASH + FEE

    p.cash = 500
    g.bank_due, g.due = 0, {1: FEE}
    g._settle(p, None)
    assert p.out and p.cash == 0
    assert g.players[1].cash == STARTING_CASH + FEE + 500       # gets what there was
    assert g.winner is None                             # two players are still in

    q = g.players[1]
    q.cash = 0
    g.bank_due, g.due = BANK_FEE, {}
    g._settle(q, None)
    assert q.out and g.winner == 2                      # last one left wins


# --------------------------------------------------------------- purchases


def test_default_buying_express_first_superchief_only_when_lines_are_gone(tables):
    g = Game(3, 1, tables)
    p = g.players[0]
    p.cash = 60000
    assert p.policy.choose_purchase(g, p) == EXPRESS
    p.express = True
    choice = p.policy.choose_purchase(g, p)
    assert choice is not None and choice >= 0           # a railroad, while the bank has any
    for r in range(tables.rr_count):
        g._transfer(r, 1)
    p.cash = 100000
    assert p.policy.choose_purchase(g, p) == SUPERCHIEF
    p.superchief = True
    assert p.policy.choose_purchase(g, p) is None
    p.superchief, p.cash = False, SUPERCHIEF_PRICE + 1000       # not enough left to travel on
    assert p.policy.choose_purchase(g, p) is None


def test_buying_keeps_a_reserve_and_values_lines_by_cover_lock_and_toll(tables):
    g = Game(4, 1, tables)
    p, pol = g.players[0], g.players[0].policy
    p.express = True
    p.cash = 3500                                       # below the reserve: buys nothing
    assert pol.choose_purchase(g, p) is None

    outlook = pol.outlook(g, p)
    sp, rfp = tables.rr_ids.index("SP"), tables.rr_ids.index("RF+P")
    assert pol.line_value(g, p, sp, outlook) > pol.line_value(g, p, rfp, outlook)
    # Owning the cities already makes a line worth less: no cover to gain.
    before = pol.line_value(g, p, rfp, outlook)
    p.access |= tables.rr_mask[rfp]
    assert pol.line_value(g, p, rfp, outlook) < before
    # Portland ME is reached only by the B+M: buying it locks everyone else out.
    bm = tables.rr_ids.index("B+M")
    locked = tables.rr_mask[bm] & ~outlook[3]
    assert tables.mass(locked) > 0


def test_arrival_pays_and_allows_one_purchase(tables):
    g = Game(3, 1, tables)
    p = g.players[0]
    put_on_trip(g, p, "New York", "Albany")
    ride(g, p, 12)
    pay = g.dest.payoff[g.dest.index("New York")][g.dest.index("Albany")]
    assert p.express and p.destination == -1 and p.trips == 1
    assert p.cash == STARTING_CASH + pay - EXPRESS_PRICE - BANK_FEE
    assert p.city == g.dest.index("Albany")


# -------------------------------------------------------------- bonus rolls


@pytest.mark.parametrize("dice, express, superchief, bonus", [
    (0, False, False, False),       # double ones, plain train
    (35, False, False, True),       # boxcars
    (0, True, False, True),         # doubles with an Express
    (1, True, False, False),        # no doubles
    (1, False, True, True),         # Superchief: every turn
])
def test_bonus_roll(tables, dice, express, superchief, bonus):
    g = Game(3, 1, tables, record=True)
    p = g.players[0]
    put_on_trip(g, p, "New York", "Los Angeles")
    p.express, p.superchief = express, superchief
    g.rand = scripted([(dice + 0.5) / 36, 0.5])
    g.play_turn()
    moves = [e for e in g.recording["turns"][-1]["ev"] if e["t"] == "move"]
    assert len(moves) == (2 if bonus else 1)
    assert p.step == MOVE[dice] + (4 if bonus else 0)   # the bonus die shows a 4


# ------------------------------------------------------------ destinations


class StayHome(Policy):
    def choose_region(self, game, p):
        return game.dest.region[p.city]


def test_rolling_your_own_city_loses_the_turn(tables):
    g = Game(3, 1, tables, policies=StayHome(), record=True)
    p = g.players[0]
    region = g.dest.region[p.city]
    k1 = tables.region_by_roll.index(region)
    k2 = tables.city_by_roll[region].index(p.city)
    g.rand = scripted([(k1 + 0.5) / 216, (k2 + 0.5) / 216])
    g.play_turn()
    events = g.recording["turns"][-1]["ev"]
    assert [e["t"] for e in events] == ["dest"] and events[0]["lost"]
    assert p.destination == -1 and p.cash == STARTING_CASH and g.seat == 1


# ------------------------------------------------------------ route engine


def test_route_engine_goes_round_opponents_when_it_pays(tables):
    g = Game(3, 1, tables)
    p = g.players[0]
    nyc = tables.rr_ids.index("NYC")
    put_on_trip(g, p, "New York", "Chicago", plan=True)
    assert set(p.rrs) == {nyc} and len(p.rrs) == 11     # nothing to avoid: the shortest way

    g._transfer(nyc, 1)
    put_on_trip(g, p, "New York", "Chicago", plan=True)
    assert nyc not in p.rrs and len(p.rrs) <= 15        # a short detour on bank track

    # Told that a dot of travel is worth a fortune, it pays the fee instead.
    class Hurry(Policy):
        STEP_VALUE = 100000
    p.policy = Hurry()
    put_on_trip(g, p, "New York", "Chicago", plan=True)
    assert len(p.rrs) == 11


def test_search_returns_a_real_route_and_respects_blocked_track(tables):
    d = tables.destinations
    a, b = tables.city_node[d.index("Seattle")], tables.city_node[d.index("Miami")]
    dots, runs = search(tables, a, b, -1, 800, 700)
    assert dots[0] == a and dots[-1] == b
    assert len(dots) - 1 == tables.hops[a][b] == sum(n for _, n in runs)
    for x, y in zip(dots, dots[1:]):
        assert any(edge[0] == y for edge in tables.adj[x])

    # Close the first piece of track of that route: the search must leave another way.
    first = next(edge for edge in tables.adj[a] if edge[0] == dots[1])
    blocked = {(a, dots[1], r) for r in first[1]} | {(dots[1], a, r) for r in first[1]}
    other = search(tables, a, b, -1, 800, 700, blocked)
    assert other[0][1] != dots[1]
    # Nothing beats the shortest route, so asking for better finds nothing.
    assert search(tables, a, b, -1, 800, 700, None, bound=800 * tables.hops[a][b]) is None


def test_replanning_never_reuses_track_ridden_this_trip(tables):
    g = Game(3, 1, tables)
    p = g.players[0]
    put_on_trip(g, p, "New York", "Chicago", plan=True)
    ride(g, p, 6)                                       # part way along the NYC, mid-stretch
    ridden = set(zip(p.dots, p.dots[1:p.step + 1], p.rrs))
    here = p.dot
    g._transfer(tables.rr_ids.index("NYC"), 1)
    assert p.replan
    p.est = None                                        # pretend the old fee does not apply
    g._replan(p, None)
    assert p.dots[0] == here and p.dots[-1] == tables.city_node[g.dest.index("Chicago")]
    new = set(zip(p.dots, p.dots[1:], p.rrs))
    assert not (new & ridden) and not ({(b, a, r) for a, b, r in new} & ridden)


# ------------------------------------------------------------------ winning


def test_declaring_at_home_wins_on_the_spot(tables):
    g = Game(3, 1, tables)
    p = g.players[0]
    p.cash = TARGET_CASH
    g.play_turn()
    assert g.winner == 0


class AlwaysDeclares(Policy):
    DECLARE_URGENT = 0          # any opponent at all is reason enough


def test_default_player_declares_when_home_is_near_and_waits_when_it_is_far(tables):
    g = Game(4, 1, tables)
    p, rivals = g.players[0], g.players[1:]
    t, hops = tables, tables.hops
    home = t.city_node[p.home]
    p.cash = TARGET_CASH
    typical = t.typical_hops(home) / Policy.MOVE
    cities = sorted((c for c in range(g.dest.count) if t.city_node[c] != home),
                    key=lambda c: hops[t.city_node[c]][home])

    def stand(q, node):
        q.dots, q.step = (node,), 0

    def ask(city, superchief=False):
        p.city, p.superchief = city, superchief
        return p.policy.choose_declare(g, p)

    def clear_the_way(city):
        way = t.shortest(t.city_node[city], home)[0]
        for q in rivals:
            stand(q, max(range(len(hops)), key=lambda n: min(hops[n][d] for d in way)))
        return way

    near, far = cities[0], cities[-1]
    clear_the_way(near)
    assert ask(near)                                    # closer than a typical city: go
    clear_the_way(far)
    assert not ask(far)                                 # further: another trip will end closer
    p.city = p.home
    assert p.policy.choose_declare(g, p)                # at home it is a win on the spot

    # A Superchief makes the same distance fewer turns.
    fast = next(c for c in cities if hops[t.city_node[c]][home] / Policy.MOVE > typical
                >= hops[t.city_node[c]][home] / Policy.MOVE_SUPERCHIEF)
    clear_the_way(fast)
    assert not ask(fast) and ask(fast, superchief=True)

    # An opponent near the way home halves the trip worth risking.
    risky = next(c for c in cities if typical / 2 < hops[t.city_node[c]][home] / Policy.MOVE <= typical)
    way = clear_the_way(risky)
    assert ask(risky)
    stand(rivals[0], way[len(way) // 2])
    assert not ask(risky)

    # No time to wait once an opponent is close to winning, or already declared.
    clear_the_way(far)
    rivals[1].cash = int(TARGET_CASH * Policy.DECLARE_URGENT)
    assert ask(far)
    rivals[1].cash = STARTING_CASH
    rivals[1].declared = True
    assert ask(far)


def test_declared_player_heads_home_and_wins_there(tables):
    g = Game(3, 5, tables, record=True)
    p = g.players[0]
    p.cash = TARGET_CASH + 100000
    far = max(range(g.dest.count), key=lambda c: tables.hops[tables.city_node[c]][tables.city_node[p.home]])
    p.city = far
    p.dots = (tables.city_node[far],)
    p.policy = AlwaysDeclares()
    g.play_turn()
    assert p.declared and p.goal == tables.city_node[p.home] and p.destination >= 0
    while g.winner is None and g.turn < 300:
        g.play_turn()
    assert g.winner == 0 and p.dot == tables.city_node[p.home]
    last = g.recording["turns"][-1]["ev"]
    assert last[-1] == {"t": "win", "how": "reached home", "p": 0}


def test_declared_player_who_drops_below_the_target_goes_to_the_alternate(tables):
    g = Game(3, 5, tables)
    p = g.players[0]
    p.cash = TARGET_CASH + 500
    far = max(range(g.dest.count), key=lambda c: tables.hops[tables.city_node[c]][tables.city_node[p.home]])
    p.city = far
    p.dots = (tables.city_node[far],)
    p.policy = AlwaysDeclares()
    g.play_turn()                                       # declares, moves, pays $1,000, falls short
    assert not p.declared and p.destination >= 0
    assert p.goal == tables.city_node[p.destination]
    assert p.cash == TARGET_CASH + 500 - BANK_FEE


# -------------------------------------------------------------- rover play


def declare_on_the_way_home(g, q, start, alternate, steps):
    """Make player q a declared player `steps` dots into the trip home from `start`."""
    d, t = g.dest, g.tables
    q.cash = TARGET_CASH + 60000
    q.city = q.trip_from = d.index(start)
    q.destination = d.index(alternate)
    q.declared, q.at_home, q.trail, q.est = True, False, None, None
    g.declared += 1
    q.dots, _, _, q.rrs = t.shortest(t.city_node[q.city], t.city_node[q.home])
    q.step, q.goal = steps, q.dots[-1]


def test_moving_through_a_declared_player_collects_50000_once(tables):
    g = Game(4, 1, tables)
    p, q, late = g.players[0], g.players[1], g.players[2]
    q.home = g.dest.index("Chicago")
    declare_on_the_way_home(g, q, "New York", "Miami", 4)     # on the NYC line, four dots out
    spot = q.dot

    put_on_trip(g, p, "New York", "Chicago")                    # the same line, behind q
    ride(g, p, 7)                                               # passes through q's dot
    assert spot in p.dots[1:8]
    assert p.cash == STARTING_CASH + ROVER_PRIZE - BANK_FEE
    assert q.cash == TARGET_CASH + 60000 - ROVER_PRIZE
    assert not q.declared and g.declared == 0
    assert q.goal == tables.city_node[g.dest.index("Miami")]    # sent to the alternate
    assert q.dots[0] == spot                                    # from where they were caught

    put_on_trip(g, late, "New York", "Chicago")                 # a second pawn gets nothing
    ride(g, late, 7)
    assert late.cash == STARTING_CASH - BANK_FEE


def test_declared_player_passes_others_freely_and_landing_short_is_no_catch(tables):
    g = Game(4, 1, tables)
    p, q = g.players[0], g.players[1]
    q.home = g.dest.index("Chicago")
    declare_on_the_way_home(g, q, "New York", "Miami", 0)
    put_on_trip(g, p, "New York", "Chicago")
    ride(g, p, 3)                                               # p sits three dots up the line
    assert q.declared                                           # starting on the same dot is no catch

    g.bank_due, g.due = 0, {}
    g._move(q, 6, None, [3, 3])                                 # q runs straight through p
    assert q.declared and q.cash == TARGET_CASH + 60000 and p.cash == STARTING_CASH - BANK_FEE

    ride(g, p, 2)                                               # p moves, but stops short of q
    assert q.declared and p.cash == STARTING_CASH - 2 * BANK_FEE


def test_default_player_chases_only_when_it_is_worth_it(tables):
    g = Game(4, 1, tables)
    p, q = g.players[0], g.players[1]
    d = g.dest
    q.home = d.index("Seattle")
    declare_on_the_way_home(g, q, "Boston", "Miami", 0)         # q stands in Boston
    put_on_trip(g, p, "New York", "Chicago")                    # Boston is three dots off p's way
    target = q.dot
    assert target not in p.dots

    route = p.policy.choose_rover(g, p, q, set())
    assert route is not None
    dots, rrs = route
    assert dots[0] == p.dot and dots[-1] == p.goal and target in dots[1:13]
    tracks = [(min(a, b), max(a, b), r) for a, b, r in zip(dots, dots[1:], rrs)]
    assert len(tracks) == len(set(tracks))                      # no track used twice, out and back

    ride(g, p, 12)                                              # the engine asks, p chases and catches
    assert p.cash == STARTING_CASH + ROVER_PRIZE - BANK_FEE and not q.declared

    # Too far to reach in one roll: not worth turning aside.
    g2 = Game(4, 1, tables)
    p2, q2 = g2.players[0], g2.players[1]
    q2.home = d.index("Seattle")
    declare_on_the_way_home(g2, q2, "Miami", "Boston", 0)
    put_on_trip(g2, p2, "New York", "Chicago")
    assert p2.policy.choose_rover(g2, p2, q2, set()) is None

    # A player who puts no value on travel time's cost of chasing still will not chase out of range,
    # and one told chasing is worthless never does.
    class Indifferent(Policy):
        STOP_VALUE = -ROVER_PRIZE
    p.policy = Indifferent()
    declare_on_the_way_home(g, q, "Boston", "Miami", 0)
    put_on_trip(g, p, "New York", "Chicago")
    assert p.policy.choose_rover(g, p, q, set()) is None


# ------------------------------------------------- whole games, checked


def check_recording(tables, rec):
    """Replay a recorded game and check every event against the rules."""
    d = tables.destinations
    index = {n: i for i, n in enumerate(tables.node_ids)}
    n = len(rec["players"])
    cash = [rec["start_cash"]] * n
    owner = [BANK] * tables.rr_count
    city = [p["home"] for p in rec["players"]]
    dot = [tables.city_node[c] for c in city]
    dest, route, step, trail = [None] * n, [None] * n, [0] * n, [set() for _ in range(n)]
    express, superchief, declared, out = [False] * n, [False] * n, [False] * n, [False] * n
    all_sold = False
    turned_back = [False] * n
    assert len({tables.city_place[c] for c in city}) == n

    for turn in rec["turns"]:
        i = turn["p"]
        assert not out[i]
        ridden, pay, spent, to_bank, from_bank, bust, eligible = [], 0, 0, 0, 0, False, None
        passed, owed_catch, caught = [], set(), set()
        rate = FEE_ALL_SOLD if all_sold else FEE
        for e in turn["ev"]:
            kind = e["t"]
            if kind in ("route", "arrive", "buy") and e.get("p", i) != i:
                # Another player's business inside this turn: a caught player
                # sets off for, or is already at, their alternate destination.
                j = e["p"]
                if kind == "route":
                    dots = [index[x] for x in e["nodes"]]
                    assert dots[0] == dot[j] and dots[-1] == tables.city_node[dest[j]]
                    route[j], step[j] = (dots, e["rrs"]), 0
                elif kind == "arrive":
                    assert e["city"] == dest[j] and e["pay"] == d.payoff[city[j]][dest[j]]
                    cash[j] += e["pay"]
                    city[j], dest[j] = dest[j], None
                else:
                    cash[j] -= e["price"]
                    if e["item"] == "line":
                        owner[e["rr"]] = j
                        all_sold = all_sold or BANK not in owner
                    elif e["item"] == "express":
                        express[j] = True
                    else:
                        superchief[j] = True
                continue
            if kind == "rover":
                dots = [index[x] for x in e["nodes"]]
                assert declared[e["target"]] and not declared[i]
                assert dots[0] == dot[i] and dots[-1] == tables.city_node[dest[i]]
                assert dot[e["target"]] in dots[1:13]                           # by way of the declared pawn
                seen = set(trail[i])
                for a, b, r in zip(dots, dots[1:], e["rrs"]):
                    assert any(edge[0] == b and r in edge[1] for edge in tables.adj[a])
                    assert (a, b, r) not in seen and (b, a, r) not in seen       # once per trip, both legs
                    seen.add((a, b, r))
                route[i], step[i] = (dots, e["rrs"]), 0
            elif kind == "caught":
                q = e["p"]
                assert e["by"] == i and declared[q] and dot[q] in passed
                assert e["pay"] == ROVER_PRIZE
                cash[q] -= e["pay"]
                cash[i] += e["pay"]
                owed_catch.discard(q)
                caught.add(q)
            elif kind == "dest":
                assert dest[i] is None
                assert d.region[e["city"]] == e["region"]
                assert tables.region_by_roll[e["region_roll"]] == e["rolled"]
                assert tables.city_by_roll[e["region"]][e["city_roll"]] == e["city"]
                if e["rolled"] != e["region"]:
                    assert e["rolled"] == d.region[city[i]]             # only your own region is a choice
                assert e["lost"] == (tables.city_place[e["city"]] == tables.city_place[city[i]])
                if not e["lost"]:
                    dest[i], declared[i], trail[i] = e["city"], e["declared"], set()
                    turned_back[i] = False
                    if e["declared"]:
                        assert cash[i] >= rec["target_cash"]
            elif kind == "route":
                dots = [index[x] for x in e["nodes"]]
                assert dots[0] == dot[i]
                goal = rec["players"][i]["home"] if declared[i] else dest[i]
                assert dots[-1] == tables.city_node[goal]
                assert len(e["rrs"]) == len(dots) - 1
                for a, b, r in zip(dots, dots[1:], e["rrs"]):
                    assert any(edge[0] == b and r in edge[1] for edge in tables.adj[a])
                    # Once per trip. The one exception in the rules: a player turned back
                    # from the trip home may re-use track if there is no other way on.
                    assert turned_back[i] or ((a, b, r) not in trail[i] and (b, a, r) not in trail[i])
                route[i], step[i] = (dots, e["rrs"]), 0
            elif kind == "move":
                dots, rrs = route[i]
                assert e["from"] == step[i]
                roll = sum(e["dice"])
                assert e["to"] == min(step[i] + roll, len(rrs))                 # the whole roll, or arrive
                if e["bonus"]:
                    assert eligible                                             # earned on the white dice
                else:
                    a, b = e["dice"]
                    eligible = superchief[i] or (a == b and (express[i] or a == 6))
                for k in range(e["from"], e["to"]):
                    ridden.append(owner[rrs[k]])
                    trail[i].add((dots[k], dots[k + 1], rrs[k]))
                passed = dots[e["from"] + 1:e["to"] + 1]
                owed_catch |= {q for q in range(n) if q != i and declared[q] and dot[q] in passed}
                step[i] = e["to"]
                dot[i] = dots[step[i]]
            elif kind == "arrive":
                assert dot[i] == tables.city_node[e["city"]] and e["city"] == dest[i]
                assert e["from"] == city[i] and e["pay"] == d.payoff[city[i]][dest[i]]
                pay += e["pay"]
                cash[i] += e["pay"]
                city[i], dest[i] = dest[i], None
            elif kind == "home":
                assert declared[i] and dot[i] == tables.city_node[rec["players"][i]["home"]]
            elif kind == "buy":
                price = e["price"]
                assert cash[i] >= price
                if e["item"] == "line":
                    assert owner[e["rr"]] == BANK and price == tables.rr_price[e["rr"]]
                    owner[e["rr"]] = i
                    all_sold = all_sold or BANK not in owner
                elif e["item"] == "express":
                    assert not express[i] and price == EXPRESS_PRICE
                    express[i] = True
                else:
                    assert not superchief[i] and price == SUPERCHIEF_PRICE and BANK not in owner
                    superchief[i] = True
                cash[i] -= price
                spent += price
            elif kind == "sale":
                r, half = e["rr"], tables.rr_price[e["rr"]] // 2
                assert owner[r] == i and e["price"] >= half and e["price"] % 500 == 0
                owner[r] = e["to"]
                cash[i] += e["price"]
                if e["to"] == BANK:
                    assert e["price"] == half
                    from_bank += half
                else:
                    assert cash[e["to"]] >= e["price"] and not out[e["to"]]
                    cash[e["to"]] -= e["price"]
            elif kind == "declare":
                assert cash[i] >= rec["target_cash"]
                assert tables.city_place[city[i]] == tables.city_place[rec["players"][i]["home"]]
                declared[i] = True
            elif kind == "fees":
                paid = dict(e["to"])
                required = e.get("owed", e)
                owed_to = dict(required["to"])
                assert set(owed_to) == {o for o in ridden if o not in (BANK, i)}
                assert all(a in (BANK_FEE, FEE, FEE_ALL_SOLD) for a in owed_to.values())
                assert required["bank"] == (BANK_FEE if any(o in (BANK, i) for o in ridden) else 0)
                owed = required["bank"] + sum(owed_to.values())
                bust = cash[i] < owed
                assert all(0 <= a <= owed_to[o] for o, a in paid.items())
                assert sum(paid.values()) + e["bank"] == min(cash[i], owed)
                cash[i] -= e["bank"] + sum(paid.values())
                to_bank += e["bank"]
                for o, a in paid.items():
                    cash[o] += a
            elif kind == "out":
                assert bust and i not in owner              # nothing left to sell
                out[i] = True
            elif kind == "undeclare":
                assert declared[e["p"]] and (cash[e["p"]] < rec["target_cash"] or e["p"] in caught)
                declared[e["p"]] = False
                turned_back[e["p"]] = True
            elif kind == "win":
                w = e.get("p", i)
                assert rec["winner"] == w
            else:
                raise AssertionError(f"unknown event {kind}")
        assert not owed_catch                           # every pawn passed over was caught
        assert cash == turn["cash"]
        assert not ridden or any(e["t"] == "fees" for e in turn["ev"])
    assert rec["winner"] is not None
    return owner


@pytest.mark.parametrize("players", [3, 4, 5, 6])
def test_recorded_games_follow_the_rules(tables, players):
    for seed in range(6):
        g = Game(players, 100 * players + seed, tables, record=True).play()
        owner = check_recording(tables, g.recording)
        assert owner == g.owner
        w = g.players[g.winner]
        others_out = all(p.out for p in g.players if p is not w)
        assert others_out or (w.cash >= TARGET_CASH and w.dot == tables.city_node[w.home])


def test_same_seed_plays_the_same_game_recorded_or_not(tables):
    fast = Game(4, 42, tables).play()
    slow = Game(4, 42, tables, record=True).play()
    assert fast.turn == slow.turn and fast.winner == slow.winner
    assert [p.cash for p in fast.players] == [p.cash for p in slow.players]
    assert fast.owner == slow.owner


def test_simulation_summary_and_speed(tables):
    s = simulate(150, 4, seed=5, tables=tables)
    assert s["unfinished"] == 0
    assert sum(s["win_share_by_seat"]) == pytest.approx(1.0)
    assert s["average_railroads_sold"] > 20             # players do buy
    assert s["games_per_second"] > 15


def test_workers_give_the_same_result(tables):
    one = simulate(60, 3, seed=9, tables=tables)
    many = simulate(60, 3, seed=9, tables=tables, workers=2)
    for key in ("average_rounds", "win_share_by_seat", "average_railroads_sold", "bankruptcies_per_game"):
        assert one[key] == pytest.approx(many[key])


def test_policies_can_differ_by_seat(tables):
    class NeverBuys(Policy):
        def choose_purchase(self, game, p):
            return None

    s = simulate(200, 4, seed=3, tables=tables, policies=[NeverBuys(), Policy(), Policy(), Policy()])
    assert s["win_share_by_seat"][0] < 0.15             # owning nothing loses


def test_record_games_covers_every_player_count(tables):
    games = record_games(seed=1, tables=tables)
    assert [len(g["players"]) for g in games] == [3, 4, 5, 6]
    assert all(g["winner"] is not None for g in games)
