import random
from collections import Counter

import pytest

from railbaron import REGION_PROBABILITY, Board
from railbaron.destinations import OUTCOMES, REGIONS, Destinations, outcome, roll_dice


@pytest.fixture(scope="module")
def dest():
    return Destinations.load()


def test_three_dice_give_72_equally_likely_outcomes():
    seen = Counter(outcome(b, w1, w2) for b in range(1, 7) for w1 in range(1, 7) for w2 in range(1, 7))
    assert sorted(seen) == list(range(OUTCOMES))
    assert set(seen.values()) == {3}
    rand = random.Random(3).random
    faces = [roll_dice(rand) for _ in range(2000)]
    assert {f for roll in faces for f in roll} == {1, 2, 3, 4, 5, 6}


def test_table_lookups_match_the_printed_table(dest):
    # Region roll: odd black die with a white total of 8 is the Northeast.
    assert REGIONS[dest.region_table[outcome(1, 3, 5)]] == "NE"
    assert REGIONS[dest.region_table[outcome(4, 6, 6)]] == "NW"
    ne, sw = REGIONS.index("NE"), REGIONS.index("SW")
    assert dest.names[dest.city_table[ne][outcome(5, 2, 3)]] == "Albany"
    assert dest.names[dest.city_table[sw][outcome(2, 1, 1)]] == "Los Angeles"


def test_odds_agree_with_the_board(dest):
    board = Board.load()
    assert dest.region_odds() == pytest.approx([REGION_PROBABILITY[r] for r in REGIONS])
    by_dot = Counter()
    for city, p in enumerate(dest.destination_odds()):
        by_dot[dest.nodes[city]] += p
    for city in board.cities.values():
        assert by_dot[city.node] == pytest.approx(board.destination_probability(city.name), abs=1e-5)


def test_payoff_chart(dest):
    ny, la = dest.index("New York"), dest.index("Los Angeles")
    assert dest.payoff[ny][la] == dest.payoff[la][ny] == 31000
    assert dest.index("Washington") == dest.index("Washington DC")
    assert dest.index("St Louis") == dest.index("St. Louis")
    assert all(dest.payoff[c][c] == 0 for c in range(dest.count))


def test_rolls_follow_the_exact_odds(dest):
    rand = random.Random(7).random
    chicago = dest.index("Chicago")
    n = 200_000
    counts = Counter(dest.roll_destination(chicago, None, rand) for _ in range(n))
    assert counts[chicago] > 0   # your own city can come up: that is a lost turn
    for city, p in enumerate(dest.destination_odds(chicago)):
        assert counts[city] / n == pytest.approx(p, abs=0.004)
    assert sum(dest.destination_odds(chicago)) == pytest.approx(1.0)


def test_region_choice_moves_home_region_rolls(dest):
    chicago = dest.index("Chicago")
    choice = dest.richest_region
    assert REGIONS[choice[chicago]] == "SW"
    odds = dest.destination_odds(chicago, choice)
    in_region = lambda r: sum(p for c, p in enumerate(odds) if REGIONS[dest.region[c]] == r)
    assert in_region("NC") == 0
    assert in_region("SW") == pytest.approx((12 + 11) / 72)
    assert dest.expected_payoff(chicago, choice) > dest.expected_payoff(chicago)
    rand = random.Random(1).random
    rolls = {dest.region[dest.roll_destination(chicago, choice, rand)] for _ in range(5000)}
    assert REGIONS.index("NC") not in rolls


def test_twin_cities_share_a_dot_and_pay_nothing_between_them(dest):
    mpls, stp = dest.index("Minneapolis"), dest.index("St. Paul")
    assert dest.dot[mpls] == dest.dot[stp]
    assert dest.payoff[mpls][stp] == 0
    assert dest.destination_odds(mpls)[stp] > 0


def test_batches_match_exact_odds(dest):
    chicago = dest.index("Chicago")
    rolls = dest.sample(300_000, chicago, seed=5)
    counts = Counter(rolls.tolist())
    for city, p in enumerate(dest.destination_odds(chicago)):
        assert counts[city] / 300_000 == pytest.approx(p, abs=0.003)

    cities, payoffs = dest.sample_trips(20_000, 30, seed=5)
    assert cities.shape == payoffs.shape == (20_000, 30)
    assert payoffs[0, 1] == dest.payoff[cities[0, 0]][cities[0, 1]]
    _, per_trip = dest.long_run()
    completed = payoffs[:, 10:]
    assert completed[completed > 0].mean() == pytest.approx(per_trip, rel=0.01)
