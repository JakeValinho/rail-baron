"""Destination dice, the destination table and the payoff chart.

How the game rolls a destination
--------------------------------
Three dice are thrown: one black die, read only as odd or even, and two
white dice, read as their sum (2 to 12). That is 2 x 36 = 72 equally likely
outcomes. The roll is looked up once for a region, then thrown again and
looked up in that region's table for a city.

Everything here is a table lookup on small integers, so it is quick enough
to sit in the inner loop of a simulation:

    region  0..6    index into REGIONS
    city    0..66   index into Destinations.names
    outcome 0..71   one of the 72 dice outcomes, see outcome()

Three ways to use it
--------------------
One roll at a time, for the game engine:   roll_destination()
Exact odds, with no dice at all:           destination_odds(), expected_payoff()
Huge batches with numpy:                   sample(), sample_trips()

    python -m railbaron.destinations       prints the odds and a speed test

Assumptions to check against your rulebook
------------------------------------------
* Rolling the region you are already in lets you pick any region. Pass a
  `region_choice` table to say which one; without it the roll stands.
* Rolling the city you are already in means the city dice are thrown again.
  Minneapolis and St. Paul share a dot, as do San Francisco and Oakland, so
  rolling your twin counts as rolling your own city.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Callable, Optional, Sequence

from .board import DEFAULT_DATA_DIR, Board

REGIONS = ("NE", "SE", "NC", "SC", "PL", "NW", "SW")
REGION_NAMES = {
    "NorthEast": "NE", "SouthEast": "SE", "NorthCentral": "NC", "SouthCentral": "SC",
    "Plains": "PL", "NorthWest": "NW", "SouthWest": "SW",
}
OUTCOMES = 72

# Destination names that are spelled differently on the board.
_BOARD_NAME = {
    "minneapolis": "Minneapolis/St. Paul",
    "st paul": "Minneapolis/St. Paul",
    "oakland": "San Francisco/Oakland",
    "san francisco": "San Francisco/Oakland",
    "washington": "Washington DC",
}

Rand = Callable[[], float]


def _norm(name: str) -> str:
    return name.replace(".", "").strip().lower()


def outcome(black: int, white1: int, white2: int) -> int:
    """Turn three dice faces (1-6 each) into an outcome number 0..71."""
    return (36 if black % 2 == 0 else 0) + (white1 - 1) * 6 + (white2 - 1)


def roll_dice(rand: Rand = random.random) -> tuple[int, int, int]:
    """Throw the black die and the two white dice. Returns the three faces."""
    k = int(rand() * 216)
    return k // 36 + 1, k // 6 % 6 + 1, k % 6 + 1


class Destinations:
    """The destination table and payoff chart, as lookup tables."""

    def __init__(self, names, nodes, regions, region_table, city_table, payoff):
        self.names: tuple[str, ...] = tuple(names)
        self.nodes: tuple[str, ...] = tuple(nodes)          # board dot of each city
        self.region: tuple[int, ...] = tuple(regions)       # region of each city
        self.region_table: tuple[int, ...] = tuple(region_table)        # [outcome] -> region
        self.city_table: tuple[tuple[int, ...], ...] = tuple(map(tuple, city_table))  # [region][outcome] -> city
        self.payoff: tuple[tuple[int, ...], ...] = tuple(map(tuple, payoff))  # [from][to] -> dollars

        self.count = len(self.names)
        self._index = {_norm(n): i for i, n in enumerate(self.names)}
        self._index.setdefault("washington", self._index.get("washington dc"))
        # Cities that share a dot get the same number here.
        first = {}
        self.dot: tuple[int, ...] = tuple(first.setdefault(n, i) for i, n in enumerate(self.nodes))

        # Exact odds, worked out once.
        self._region_odds = tuple(self.region_table.count(r) / OUTCOMES for r in range(len(REGIONS)))
        self._city_odds = tuple(
            tuple(table.count(c) / OUTCOMES for c in range(self.count)) for table in self.city_table
        )
        self.richest_region: tuple[int, ...] = tuple(
            max(range(len(REGIONS)), key=lambda r: self.expected_payoff_in_region(c, r))
            for c in range(self.count)
        )

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, data_dir: Path | str = DEFAULT_DATA_DIR, board: Optional[Board] = None) -> "Destinations":
        data_dir = Path(data_dir)
        board = board or Board.load(data_dir)

        with open(data_dir / "Payout.csv", newline="", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
        chart_names = [_norm(n) for n in rows[0][1:]]
        if chart_names != [_norm(r[0]) for r in rows[1:]]:
            raise ValueError("Payout.csv rows and columns are not the same cities in the same order")
        chart = {
            a: {b: round(float(v) * 1000) for b, v in zip(chart_names, row[1:])}
            for a, row in zip(chart_names, rows[1:])
        }

        with open(data_dir / "rail_baron_destinations.csv", newline="", encoding="utf-8-sig") as f:
            table_rows = list(csv.DictReader(f))

        # Cities are numbered in the order the destination table lists them.
        names, regions = [], []
        index: dict[str, int] = {}
        for row in table_rows:
            if row["stage"] == "city" and _norm(row["result"]) not in index:
                index[_norm(row["result"])] = len(names)
                names.append(row["result"])
                regions.append(REGIONS.index(REGION_NAMES[row["region"]]))

        def chart_key(name: str) -> str:
            key = _norm(name)
            return key if key in chart else key.removesuffix(" dc")

        missing = [n for n in names if chart_key(n) not in chart]
        if missing:
            raise ValueError(f"cities in the destination table but not in Payout.csv: {missing}")
        nodes = [board.node(_BOARD_NAME.get(_norm(n), n)) for n in names]
        payoff = [[chart[chart_key(a)][chart_key(b)] for b in names] for a in names]

        # Spread each table row over the dice outcomes that produce it.
        region_table = [-1] * OUTCOMES
        city_table = [[-1] * OUTCOMES for _ in REGIONS]
        for row in table_rows:
            total, even = int(row["dice_sum"]), row["black_die_parity"] == "even"
            if row["stage"] == "region":
                table, value = region_table, REGIONS.index(REGION_NAMES[row["result"]])
            else:
                table = city_table[REGIONS.index(REGION_NAMES[row["region"]])]
                value = index[_norm(row["result"])]
            for w1 in range(1, 7):
                w2 = total - w1
                if 1 <= w2 <= 6:
                    table[outcome(2 if even else 1, w1, w2)] = value
        if -1 in region_table or any(-1 in t for t in city_table):
            raise ValueError("rail_baron_destinations.csv does not cover every dice roll")

        return cls(names, nodes, regions, region_table, city_table, payoff)

    def index(self, city: str) -> int:
        """City number for a name. Accepts the table's or the chart's spelling."""
        i = self._index.get(_norm(city))
        if i is None:
            raise KeyError(f"{city!r} is not a destination")
        return i

    # ------------------------------------------------- one roll at a time

    def roll_region(self, rand: Rand = random.random) -> int:
        return self.region_table[int(rand() * OUTCOMES)]

    def roll_city(self, region: int, rand: Rand = random.random) -> int:
        return self.city_table[region][int(rand() * OUTCOMES)]

    def roll_destination(
        self,
        current: Optional[int] = None,
        region_choice: Optional[Sequence[int]] = None,
        rand: Rand = random.random,
    ) -> int:
        """Roll a new destination for a player standing at city `current`.

        `region_choice[current]` is the region the player takes when the
        dice give the region they are already in, for example
        `destinations.richest_region`. Leave it out and the roll stands.
        """
        region = self.region_table[int(rand() * OUTCOMES)]
        if current is None:
            return self.city_table[region][int(rand() * OUTCOMES)]
        if region_choice is not None and region == self.region[current]:
            region = region_choice[current]
        table, dot, here = self.city_table[region], self.dot, self.dot[current]
        while True:
            city = table[int(rand() * OUTCOMES)]
            if dot[city] != here:
                return city

    # ----------------------------------------------------------- exact odds

    def region_odds(self) -> tuple[float, ...]:
        """Chance of each region on the region roll."""
        return self._region_odds

    def city_odds(self, region: int, current: Optional[int] = None) -> list[float]:
        """Chance of each city once `region` is settled, rerolling `current`."""
        odds = list(self._city_odds[region])
        if current is not None:
            here = self.dot[current]
            blocked = sum(p for c, p in enumerate(odds) if self.dot[c] == here)
            odds = [0.0 if self.dot[c] == here else p / (1.0 - blocked) for c, p in enumerate(odds)]
        return odds

    def destination_odds(
        self, current: Optional[int] = None, region_choice: Optional[Sequence[int]] = None
    ) -> list[float]:
        """Chance of each city being the next destination."""
        weight = list(self._region_odds)
        if current is not None and region_choice is not None:
            home, pick = self.region[current], region_choice[current]
            if pick != home:
                weight[pick] += weight[home]
                weight[home] = 0.0
        odds = [0.0] * self.count
        for region, w in enumerate(weight):
            if w:
                for city, p in enumerate(self.city_odds(region, current)):
                    odds[city] += w * p
        return odds

    def expected_payoff_in_region(self, current: int, region: int) -> float:
        """Average payoff from `current` if the destination is in `region`."""
        pay = self.payoff[current]
        return sum(p * pay[c] for c, p in enumerate(self.city_odds(region, current)))

    def expected_payoff(self, current: int, region_choice: Optional[Sequence[int]] = None) -> float:
        """Average payoff of the next trip out of `current`."""
        pay = self.payoff[current]
        return sum(p * pay[c] for c, p in enumerate(self.destination_odds(current, region_choice)))

    def transition_matrix(self, region_choice: Optional[Sequence[int]] = None):
        """numpy array T where T[a][b] is the chance that a trip from a goes to b."""
        import numpy as np

        return np.array([self.destination_odds(c, region_choice) for c in range(self.count)])

    def long_run(self, region_choice: Optional[Sequence[int]] = None) -> tuple[list[float], float]:
        """Where a player spends the game, and what an average trip pays.

        Returns (share of trips ending at each city, average payoff per trip)
        once the starting city no longer matters.
        """
        import numpy as np

        T = self.transition_matrix(region_choice)
        visits = np.full(self.count, 1.0 / self.count)
        for _ in range(200):
            visits = visits @ T
        per_trip = float(visits @ (T * np.array(self.payoff)).sum(axis=1))
        return visits.tolist(), per_trip

    # ---------------------------------------------------------- big batches

    def sample(self, n: int, current: Optional[int] = None,
               region_choice: Optional[Sequence[int]] = None, seed=None):
        """n independent destination rolls from one city, as a numpy array."""
        import numpy as np

        rng = np.random.default_rng(seed)
        edges = np.cumsum(self.destination_odds(current, region_choice))
        return np.searchsorted(edges / edges[-1], rng.random(n), side="right")

    def sample_trips(self, players: int, trips: int, start: Optional[int] = None,
                     region_choice: Optional[Sequence[int]] = None, seed=None):
        """Roll `trips` destinations in a row for many players at once.

        Each player starts at `start` (or at a rolled home city) and every
        new destination is rolled from the previous one. Returns two numpy
        arrays of shape (players, trips): the cities and the payoffs.
        """
        import numpy as np

        rng = np.random.default_rng(seed)
        # One sorted list holds every city's odds: row a covers a..a+1, so a
        # single search finds each player's next city.
        edges = np.cumsum(self.transition_matrix(region_choice), axis=1)
        edges = (edges / edges[:, -1:] + np.arange(self.count)[:, None]).ravel()
        pay = np.array(self.payoff)
        if start is None:
            here = rng.choice(self.count, size=players, p=self.destination_odds())
        else:
            here = np.full(players, start)
        cities = np.empty((players, trips), dtype=np.int16)
        payoffs = np.empty((players, trips), dtype=np.int32)
        for t in range(trips):
            there = np.searchsorted(edges, rng.random(players) + here, side="right") - here * self.count
            cities[:, t], payoffs[:, t] = there, pay[here, there]
            here = there
        return cities, payoffs


def main() -> None:
    import time

    d = Destinations.load()
    print("Region odds")
    for r, p in enumerate(d.region_odds()):
        print(f"  {REGIONS[r]}  {p:6.2%}")

    print("\nMost likely destinations (from a fresh start)")
    odds = d.destination_odds()
    for c in sorted(range(d.count), key=odds.__getitem__, reverse=True)[:8]:
        print(f"  {d.names[c]:16} {odds[c]:6.2%}")

    _, plain = d.long_run()
    _, smart = d.long_run(d.richest_region)
    print(f"\nAverage payoff per trip: ${plain:,.0f}")
    print(f"  choosing the richest region when allowed: ${smart:,.0f}")

    n = 1_000_000
    rand = random.Random(1).random
    roll, chicago = d.roll_destination, d.index("Chicago")
    t = time.perf_counter()
    for _ in range(n):
        roll(chicago, None, rand)
    one = time.perf_counter() - t
    t = time.perf_counter()
    d.sample(n, chicago, seed=1)
    batch = time.perf_counter() - t
    t = time.perf_counter()
    d.sample_trips(10_000, 100, seed=1)
    chain = time.perf_counter() - t
    print(f"\nSpeed for {n:,} destination rolls")
    print(f"  one at a time (game engine): {one:.2f} s")
    print(f"  numpy batch from one city:   {batch:.3f} s")
    print(f"  10,000 players x 100 trips:  {chain:.3f} s")


if __name__ == "__main__":
    main()
