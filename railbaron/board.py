"""The Rail Baron board as a NetworkX graph.

This module knows the map and nothing else: dots, track, railroads, cities.
Turn order, money, dice and AI decisions belong in the game engine, which
asks this module questions such as "how far is it", "what does this route
cost me" and "where can I end up with this roll".

Vocabulary
----------
node       a dot on the board, e.g. "NC_38". Cities are dots with a name.
track      one railroad's line between two adjacent dots. Two railroads
           running side by side between the same dots are two tracks.
ownership  a mapping {railroad short name: owner}. A railroad that is
           missing, or mapped to None, belongs to the bank.

Anywhere a node is expected you may pass a city name ("Chicago") instead.
"""

from __future__ import annotations

import csv
import heapq
import json
from collections import deque
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Hashable, Iterable, Mapping, Optional

import networkx as nx

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "Map Data"

# Chance that a destination roll lands in each region, out of the 72 dice
# outcomes. A test checks these against rail_baron_destinations.csv.
REGION_PROBABILITY: dict[str, float] = {
    "NE": 15 / 72,
    "SE": 9 / 72,
    "NC": 11 / 72,
    "SC": 9 / 72,
    "PL": 8 / 72,
    "NW": 8 / 72,
    "SW": 12 / 72,
}

Ownership = Mapping[str, Optional[Hashable]]
Track = tuple[str, str, str]  # (node, node, railroad) with the nodes sorted


@dataclass(frozen=True)
class Railroad:
    short_name: str
    name: str
    price: int


@dataclass(frozen=True)
class City:
    name: str
    region: str
    node: str
    probability: float  # chance of this city once its region is rolled
    payout_number: int  # row/column of this city in the payoff chart


@dataclass(frozen=True)
class FeeModel:
    """What one turn of riding costs, by who owns the track.

    Bank track costs `bank` once per turn no matter how many bank railroads
    are used. Each opponent whose track is used is paid `opponent` once per
    turn. Raise `opponent` to 10000 once every railroad has been sold.
    """

    own: int = 0
    bank: int = 1000
    opponent: int = 5000

    def for_owner(self, owner: Optional[Hashable], player: Hashable) -> int:
        if owner is None:
            return self.bank
        return self.own if owner == player else self.opponent


@dataclass(frozen=True)
class Route:
    """A walk across the board. railroads[i] carries nodes[i] to nodes[i+1]."""

    nodes: tuple[str, ...]
    railroads: tuple[str, ...]

    @property
    def distance(self) -> int:
        return len(self.railroads)

    @property
    def start(self) -> str:
        return self.nodes[0]

    @property
    def end(self) -> str:
        return self.nodes[-1]

    @property
    def railroads_used(self) -> tuple[str, ...]:
        """Distinct railroads in the order they are first ridden."""
        return tuple(dict.fromkeys(self.railroads))

    @property
    def tracks(self) -> tuple[Track, ...]:
        return tuple(
            track_key(a, b, rr)
            for a, b, rr in zip(self.nodes, self.nodes[1:], self.railroads)
        )


def track_key(a: str, b: str, railroad: str) -> Track:
    return (a, b, railroad) if a <= b else (b, a, railroad)


class Board:
    """The map. Build one with Board.load() and share it; it is read-only."""

    def __init__(
        self,
        graph: nx.MultiGraph,
        railroads: dict[str, Railroad],
        cities: dict[str, City],
        region_probability: Mapping[str, float] = REGION_PROBABILITY,
    ):
        self.graph = graph
        self.railroads = railroads
        self.cities = cities
        self.region_probability = dict(region_probability)
        self._city_by_node = {c.node: c for c in cities.values()}
        self._city_lookup = {name.lower(): c.node for name, c in cities.items()}

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, data_dir: Path | str = DEFAULT_DATA_DIR, **kwargs) -> "Board":
        """Read nodes.csv, edges.csv, cities.csv and the railroad list."""
        data_dir = Path(data_dir)

        with open(data_dir / "rail_baron_graph.json", encoding="utf-8") as f:
            railroads = {
                r["short_name"]: Railroad(r["short_name"], r["name"], int(r["price"]))
                for r in json.load(f)["railroads"]
            }

        graph = nx.MultiGraph()
        for row in _read_csv(data_dir / "nodes.csv"):
            graph.add_node(
                row["id"],
                region=row["region"],
                x=int(row["x"]),
                y=int(row["y"]),
                city=row["city"] or None,
            )

        for row in _read_csv(data_dir / "edges.csv"):
            a, b, rr = row["from"], row["to"], row["railroad"]
            if a not in graph or b not in graph:
                raise ValueError(f"edge {row['id']} references an unknown node")
            if rr not in railroads:
                raise ValueError(f"edge {row['id']} references unknown railroad {rr}")
            # The railroad is the edge key, so parallel tracks stay distinct.
            graph.add_edge(a, b, key=rr, railroad=rr, id=row["id"])

        # Dots that no track touches cannot be visited; drop them.
        graph.remove_nodes_from([n for n in graph if graph.degree(n) == 0])

        cities = {}
        for row in _read_csv(data_dir / "cities.csv"):
            if row["node_id"] not in graph:
                raise ValueError(f"city {row['name']} is not on any track")
            cities[row["name"]] = City(
                name=row["name"],
                region=row["region"],
                node=row["node_id"],
                probability=float(row["probability"]) / 100.0,
                payout_number=int(row["payout_number"]),
            )

        return cls(graph, railroads, cities, **kwargs)

    # --------------------------------------------------------------- lookups

    def node(self, where: str) -> str:
        """Turn a city name or a node id into a node id."""
        if where in self.graph:
            return where
        try:
            return self._city_lookup[where.lower()]
        except KeyError:
            raise KeyError(f"{where!r} is neither a node id nor a city name") from None

    def city_at(self, node: str) -> Optional[City]:
        return self._city_by_node.get(node)

    def label(self, node: str) -> str:
        city = self.city_at(node)
        return city.name if city else node

    def railroads_between(self, a: str, b: str) -> tuple[str, ...]:
        return tuple(self.simple[a][b]["railroads"])

    def railroads_at(self, where: str) -> set[str]:
        node = self.node(where)
        return {rr for _, _, rr in self.graph.edges(node, keys=True)}

    def destination_probability(self, city: str) -> float:
        """Chance that a random destination roll is this city."""
        c = self.cities[city]
        return self.region_probability[c.region] * c.probability

    @cached_property
    def simple(self) -> nx.Graph:
        """One edge per pair of adjacent dots, listing the railroads on it."""
        g = nx.Graph()
        g.add_nodes_from(self.graph.nodes(data=True))
        for a, b, rr in self.graph.edges(keys=True):
            if g.has_edge(a, b):
                g[a][b]["railroads"].append(rr)
            else:
                g.add_edge(a, b, railroads=[rr])
        return g

    @cached_property
    def _distances(self) -> dict[str, dict[str, int]]:
        return dict(nx.all_pairs_shortest_path_length(self.simple))

    # ------------------------------------------------------- shortest routes

    def distance(self, a: str, b: str) -> int:
        """Minimum number of dots moved to get from a to b."""
        return self._distances[self.node(a)][self.node(b)]

    def shortest_route(
        self,
        a: str,
        b: str,
        ownership: Optional[Ownership] = None,
        player: Hashable = None,
        fees: FeeModel = FeeModel(),
    ) -> Route:
        """Fewest moves from a to b.

        When several routes tie on length, ownership (if given) breaks the
        tie toward the one with the lowest fees for `player`.
        """
        # A step costs more than any possible saving in fees, so length wins.
        big = 1 + self.graph.number_of_edges() * max(fees.own, fees.bank, fees.opponent, 1)
        return self._route(a, b, ownership or {}, player, fees, step_cost=big, moves_per_turn=1)

    def cheapest_route(
        self,
        a: str,
        b: str,
        ownership: Ownership,
        player: Hashable,
        fees: FeeModel = FeeModel(),
        step_cost: float = 1.0,
        moves_per_turn: float = 7.0,
    ) -> Route:
        """Route from a to b that keeps fees paid by `player` low.

        Fees are charged per turn, and how a trip splits into turns depends
        on dice that have not been rolled yet. So each dot of track is
        charged its owner's fee divided by `moves_per_turn` (7 is the average
        of two dice). Use estimate_fee() or turn_fee() for the real charge.

        `step_cost` is what one extra dot of travel is worth in dollars. At
        the default of 1 the route avoids fees at almost any detour. Raise it
        to make the player accept fees in exchange for a faster trip.
        """
        return self._route(a, b, ownership, player, fees, step_cost, moves_per_turn)

    def _route(self, a, b, ownership, player, fees, step_cost, moves_per_turn) -> Route:
        src, dst = self.node(a), self.node(b)
        simple = self.simple

        def weight(rr: str) -> float:
            return fees.for_owner(ownership.get(rr), player) / moves_per_turn + step_cost

        # Dijkstra over dots, using the cheapest railroad on each segment.
        best = {src: 0.0}
        prev: dict[str, str] = {}
        heap = [(0.0, src)]
        done = set()
        while heap:
            cost, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            if u == dst:
                break
            for v, data in simple[u].items():
                if v in done:
                    continue
                c = cost + min(weight(rr) for rr in data["railroads"])
                if c < best.get(v, float("inf")):
                    best[v] = c
                    prev[v] = u
                    heapq.heappush(heap, (c, v))
        if dst not in done:
            raise nx.NetworkXNoPath(f"no route from {a} to {b}")

        nodes = [dst]
        while nodes[-1] != src:
            nodes.append(prev[nodes[-1]])
        nodes.reverse()

        # Where railroads tie on a segment, pick the sequence with the fewest
        # changes of railroad, so a route stays on one line where it can.
        options = []
        for u, v in zip(nodes, nodes[1:]):
            rrs = simple[u][v]["railroads"]
            low = min(weight(rr) for rr in rrs)
            options.append([rr for rr in rrs if weight(rr) <= low + 1e-9])
        return Route(tuple(nodes), _fewest_switches(options))

    # ------------------------------------------------------------------ fees

    def turn_fee(
        self,
        railroads: Iterable[str],
        ownership: Ownership,
        player: Hashable,
        fees: FeeModel = FeeModel(),
    ) -> int:
        """Fee for one turn in which `player` rode the given railroads."""
        owners = {ownership.get(rr) for rr in railroads}
        total = 0
        for owner in owners:
            total += fees.for_owner(owner, player)
        return total

    def estimate_fee(
        self,
        route: Route,
        ownership: Ownership,
        player: Hashable,
        fees: FeeModel = FeeModel(),
        moves_per_turn: int = 7,
    ) -> int:
        """Fee for a whole route if every turn moved `moves_per_turn` dots."""
        total = 0
        for i in range(0, route.distance, moves_per_turn):
            total += self.turn_fee(route.railroads[i : i + moves_per_turn], ownership, player, fees)
        return total

    # -------------------------------------------------------------- movement

    def legal_moves(
        self,
        start: str,
        roll: int,
        destination: Optional[str] = None,
        used_tracks: Iterable[Track] = (),
    ) -> list[Route]:
        """Every way to move `roll` dots from `start`.

        A move uses the whole roll, except that reaching `destination` ends
        it early. No track may be ridden twice in one trip: pass the tracks
        already ridden on earlier turns of this trip as `used_tracks`
        (Route.tracks gives them). Moves that would dead-end before the roll
        is used up are not legal and are left out.

        The list grows very fast with the roll: about 4,000 moves for a 7
        out of Chicago and 670,000 for a 12. Use reachable() when the end
        points are enough.
        """
        src = self.node(start)
        dst = self.node(destination) if destination is not None else None
        used = set(used_tracks)
        moves: list[Route] = []
        nodes, rrs = [src], []

        def walk(u: str, left: int) -> None:
            if left == 0 or (u == dst and rrs):
                moves.append(Route(tuple(nodes), tuple(rrs)))
                return
            for v, data in self.simple[u].items():
                for rr in data["railroads"]:
                    key = track_key(u, v, rr)
                    if key in used:
                        continue
                    used.add(key)
                    nodes.append(v)
                    rrs.append(rr)
                    walk(v, left - 1)
                    rrs.pop()
                    nodes.pop()
                    used.discard(key)

        if roll > 0:
            walk(src, roll)
        return moves

    def reachable(
        self,
        start: str,
        roll: int,
        destination: Optional[str] = None,
        used_tracks: Iterable[Track] = (),
    ) -> dict[str, Route]:
        """Dots where a move of `roll` from `start` can end.

        Same rules as legal_moves(). Returns {end dot: one route that gets
        there}, so `dot in result` and `set(result)` both work. The route is
        an example, not the shortest or the cheapest one. This stays
        fast for any roll, where legal_moves() does not, because it gives up
        on a branch as soon as it cannot lead anywhere new.
        """
        src = self.node(start)
        dst = self.node(destination) if destination is not None else None
        dist = self._distances
        used = set(used_tracks)
        ends: dict[str, Route] = {}
        seen = set()
        nodes, rrs = [src], []

        simple = self.simple
        hopes: dict[tuple, frozenset] = {}

        def hope(u: str, came_from: Optional[str], left: int) -> frozenset:
            """Every dot that might be an end point from here (never too few).

            This forgets which track is used up, apart from refusing to turn
            straight back along a single track, so it is quick to work out.
            """
            if left == 0 or (u == dst and came_from is not None):
                return frozenset((u,))
            key = (u, came_from, left)
            found = hopes.get(key)
            if found is None:
                parts = [
                    hope(v, u, left - 1)
                    for v, data in simple[u].items()
                    if v != came_from or len(data["railroads"]) > 1
                ]
                found = hopes[key] = frozenset().union(*parts)
            return found

        def walk(u: str, left: int) -> None:
            if left == 0 or (u == dst and rrs):
                ends.setdefault(u, Route(tuple(nodes), tuple(rrs)))
                return
            # Stop if every dot this branch could still reach is already found.
            if hope(u, nodes[-2] if rrs else None, left) <= ends.keys():
                return
            # Only used track close enough to ride again can change what
            # happens from here. Two arrivals that agree on it are the same.
            near = dist[u]
            state = (u, left, frozenset(t for t in used if near[t[0]] < left or near[t[1]] < left))
            if state in seen:
                return
            seen.add(state)
            for v, data in simple[u].items():
                for rr in data["railroads"]:
                    key = track_key(u, v, rr)
                    if key in used:
                        continue
                    used.add(key)
                    nodes.append(v)
                    rrs.append(rr)
                    walk(v, left - 1)
                    rrs.pop()
                    nodes.pop()
                    used.discard(key)
                    break  # parallel railroads lead to the same dots

        if roll > 0:
            walk(src, roll)
        return ends

    # ------------------------------------------------------ railroad values

    def railroad_graph(self, railroads: Iterable[str]) -> nx.Graph:
        """The part of the map covered by the given railroads."""
        wanted = set(railroads)
        g = nx.Graph()
        for a, b, rr in self.graph.edges(keys=True):
            if rr in wanted:
                g.add_edge(a, b)
        return g

    def cities_served(self, railroads: Iterable[str]) -> set[str]:
        nodes = self.railroad_graph(railroads).nodes
        return {c.name for c in self.cities.values() if c.node in nodes}

    def coverage(self, railroads: Iterable[str]) -> float:
        """Chance that a random destination is on one of these railroads."""
        return sum(self.destination_probability(c) for c in self.cities_served(railroads))

    def purchase_gain(self, owned: Iterable[str], candidate: str) -> float:
        """How much coverage() rises if `candidate` is added to `owned`."""
        owned = set(owned)
        return self.coverage(owned | {candidate}) - self.coverage(owned)

    @cached_property
    def traffic(self) -> dict[str, float]:
        """Expected dots ridden on each railroad during one average trip.

        A trip runs between two different cities, each picked by its
        destination probability, along a shortest route. Where shortest
        routes tie, including parallel railroads on one segment, the trip
        is split evenly between them. Ownership is ignored: this is the
        natural flow of the map, before anyone starts avoiding fees.
        """
        simple = self.simple
        dest = {c.node: self.destination_probability(c.name) for c in self.cities.values()}
        flow = dict.fromkeys(self.railroads, 0.0)

        for src, p_src in dest.items():
            # Breadth-first search counting shortest routes to every dot.
            dist = {src: 0}
            sigma = {src: 1.0}
            order = []
            queue = deque([src])
            while queue:
                u = queue.popleft()
                order.append(u)
                for v, data in simple[u].items():
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        sigma[v] = 0.0
                        queue.append(v)
                    if dist[v] == dist[u] + 1:
                        sigma[v] += sigma[u] * len(data["railroads"])

            # Walk back from the far end, pushing trip weight toward src.
            scale = p_src / (1.0 - p_src)  # the destination is never src
            through = dict.fromkeys(order, 0.0)
            for v in reversed(order):
                arriving = through[v] + (dest.get(v, 0.0) if v != src else 0.0)
                if arriving == 0.0:
                    continue
                for u, data in simple[v].items():
                    if dist[u] + 1 != dist[v]:
                        continue
                    per_track = sigma[u] / sigma[v] * arriving
                    for rr in data["railroads"]:
                        flow[rr] += per_track * scale
                        through[u] += per_track
        return flow

    def railroad_stats(self) -> list[dict]:
        """One row per railroad with the numbers a buyer cares about.

        price          purchase price
        track          dots of track
        cities         cities on the line
        coverage       chance a random destination is on the line
        monopoly       coverage from cities that no other railroad reaches
        connections    number of other railroads it meets
        traffic        expected dots ridden on it per average trip
        traffic_share  its part of all traffic on the map
        coverage_per_10k, traffic_per_10k   the same, per $10,000 of price
        Rows are sorted by traffic, busiest first.
        """
        served = {rr: self.cities_served([rr]) for rr in self.railroads}
        lines_at_city = {
            c: sum(c in cities for cities in served.values()) for c in self.cities
        }
        nodes_of = {rr: set(self.railroad_graph([rr]).nodes) for rr in self.railroads}
        total_traffic = sum(self.traffic.values())

        rows = []
        for rr, info in self.railroads.items():
            coverage = sum(self.destination_probability(c) for c in served[rr])
            monopoly = sum(
                self.destination_probability(c) for c in served[rr] if lines_at_city[c] == 1
            )
            traffic = self.traffic[rr]
            per_10k = 10000 / info.price
            rows.append(
                {
                    "railroad": rr,
                    "name": info.name,
                    "price": info.price,
                    "track": sum(1 for *_, k in self.graph.edges(keys=True) if k == rr),
                    "cities": sorted(served[rr]),
                    "coverage": coverage,
                    "monopoly": monopoly,
                    "connections": sum(
                        1 for other in self.railroads
                        if other != rr and nodes_of[rr] & nodes_of[other]
                    ),
                    "traffic": traffic,
                    "traffic_share": traffic / total_traffic,
                    "coverage_per_10k": coverage * per_10k,
                    "traffic_per_10k": traffic * per_10k,
                }
            )
        rows.sort(key=lambda r: r["traffic"], reverse=True)
        return rows


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fewest_switches(options: list[list[str]]) -> tuple[str, ...]:
    """Pick one railroad per step so that the railroad changes least often."""
    if not options:
        return ()
    cost = {rr: 0 for rr in options[0]}
    back: list[dict[str, str]] = []
    for step in options[1:]:
        cheapest = min(cost, key=cost.get)
        new_cost, pointers = {}, {}
        for rr in step:
            if rr in cost and cost[rr] <= cost[cheapest] + 1:
                new_cost[rr], pointers[rr] = cost[rr], rr
            else:
                new_cost[rr], pointers[rr] = cost[cheapest] + 1, cheapest
        cost = new_cost
        back.append(pointers)
    choice = [min(cost, key=cost.get)]
    for pointers in reversed(back):
        choice.append(pointers[choice[-1]])
    choice.reverse()
    return tuple(choice)
