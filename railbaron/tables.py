"""Everything the game engine looks up, as plain integers and tuples.

The board and destination modules are convenient but work with names. A
simulation plays thousands of games, so the engine numbers everything once
and never touches a name again:

    dot       0..540   index into node_ids
    railroad  0..27    index into rr_ids
    city      0..66    destination city, as in Destinations.names
    place     0..64    a city dot (twin cities share one place)

Sets of railroads and sets of places are often held as bit masks: bit r is
set when railroad r is in the set.

Build one Tables and share it between games; it never changes.
"""

from __future__ import annotations

import heapq
from typing import Optional

from .board import Board
from .destinations import Destinations, outcome

_INF = float("inf")


def _bits(rrs) -> int:
    mask = 0
    for r in rrs:
        mask |= 1 << r
    return mask


class Tables:
    def __init__(self, board: Optional[Board] = None, destinations: Optional[Destinations] = None):
        self.board = b = board or Board.load()
        self.destinations = d = destinations or Destinations.load(board=b)

        # ---- dice: the three destination dice have 216 face combinations.
        # Indexing by that number lets a recording show the actual dice.
        which = [outcome(k // 36 + 1, k // 6 % 6 + 1, k % 6 + 1) for k in range(216)]
        self.region_by_roll = tuple(d.region_table[o] for o in which)
        self.city_by_roll = tuple(tuple(table[o] for o in which) for table in d.city_table)

        # ---- railroads
        self.rr_ids = list(b.railroads)
        rr_index = {r: i for i, r in enumerate(self.rr_ids)}
        self.rr_name = [b.railroads[r].name for r in self.rr_ids]
        self.rr_price = [b.railroads[r].price for r in self.rr_ids]
        self.rr_count = len(self.rr_ids)

        # ---- dots and track
        self.node_ids = list(b.graph)
        index = {n: i for i, n in enumerate(self.node_ids)}
        simple = b.simple
        # adj[dot] = ((next dot, railroads on that track, length, dots in between, railroads as a mask), ...)
        adj = []
        for n in self.node_ids:
            edges = []
            for m, data in simple[n].items():
                rrs = tuple(sorted(rr_index[r] for r in data["railroads"]))
                edges.append((index[m], rrs, 1, (), _bits(rrs)))
            adj.append(tuple(edges))
        self.adj = tuple(adj)
        moves = b._distances
        self.hops = [[moves[a][c] for c in self.node_ids] for a in self.node_ids]
        self.node_rrs = [frozenset(r for edge in self.adj[n] for r in edge[1]) for n in range(len(self.node_ids))]

        # ---- cities
        self.city_node = [index[n] for n in d.nodes]
        places: dict[int, int] = {}
        self.city_place = [places.setdefault(n, len(places)) for n in self.city_node]
        self.place_count = len(places)
        self.place_node = list(places)
        self.place_prob = [0.0] * self.place_count
        for city, p in enumerate(d.destination_odds()):
            self.place_prob[self.city_place[city]] += p
        # rr_mask[r] has bit `place` set when railroad r reaches that city.
        self.rr_mask = [
            sum(1 << place for node, place in places.items() if r in self.node_rrs[node])
            for r in range(self.rr_count)
        ]

        # ---- the same track with plain stretches squeezed into one edge.
        # A dot is kept if it is a city, a junction, a dead end, or the place
        # where the set of railroads changes. Route searches run on this
        # smaller graph, which has about a third of the dots.
        city_dots = set(self.city_node)

        def keep(n: int) -> bool:
            edges = self.adj[n]
            return n in city_dots or len(edges) != 2 or edges[0][1] != edges[1][1]

        self.kadj: list = [None] * len(self.node_ids)
        for u in range(len(self.node_ids)):
            if not keep(u):
                continue
            edges = []
            for m, rrs, _, _, mask in self.adj[u]:
                inner, prev, cur = [], u, m
                while not keep(cur):
                    inner.append(cur)
                    a, c = self.adj[cur][0][0], self.adj[cur][1][0]
                    prev, cur = cur, (a if a != prev else c)
                if cur != u:
                    edges.append((cur, rrs, len(inner) + 1, tuple(inner), mask))
            self.kadj[u] = tuple(edges)

        self._shortest: dict[int, tuple] = {}
        self._mass: dict[int, float] = {}

        # toll[r]: share of trips whose shortest route has a piece of track
        # that only railroad r serves. A measure of how often others need it.
        self.toll = [0.0] * self.rr_count
        for a in range(self.place_count):
            pa = self.place_prob[a]
            for c in range(self.place_count):
                if a == c:
                    continue
                weight = pa * self.place_prob[c] / (1.0 - pa)
                runs = self.shortest(self.place_node[a], self.place_node[c])[1]
                for r in {choice[0] for choice, _ in runs if len(choice) == 1}:
                    self.toll[r] += weight

    def shortest(self, a: int, b: int) -> tuple:
        """Fewest-moves route between two dots, worked out once and remembered.

        Returns (dots, runs, first_mask, first_rrs):

        dots        the dots from a to b
        runs        the route as stretches with the same choice of railroads,
                    ((railroads, steps), ...)
        first_rrs   the railroad for each step if the first choice is always taken
        first_mask  those railroads as a bit mask. If none belongs to an
                    opponent, the route can be ridden as it stands.
        """
        key = a * 1024 + b
        found = self._shortest.get(key)
        if found is None:
            to_b, adj = self.hops[b], self.adj
            n, dots, runs, rrs, mask = a, [a], [], [], 0
            while n != b:
                want = to_b[n] - 1
                for edge in adj[n]:
                    if to_b[edge[0]] == want:
                        break
                n = edge[0]
                dots.append(n)
                choice = edge[1]
                rrs.append(choice[0])
                mask |= 1 << choice[0]
                if runs and runs[-1][0] == choice:
                    runs[-1][1] += 1
                else:
                    runs.append([choice, 1])
            found = self._shortest[key] = (tuple(dots), tuple((c, k) for c, k in runs), mask, tuple(rrs))
        return found

    def mass(self, mask: int) -> float:
        """Chance that a random destination is one of the cities in `mask`."""
        total = self._mass.get(mask)
        if total is None:
            prob, total, m = self.place_prob, 0.0, mask
            while m:
                low = m & -m
                total += prob[low.bit_length() - 1]
                m ^= low
            self._mass[mask] = total
        return total


def search(t: Tables, src: int, dst: int, free: int, step: int, penalty: int, blocked=None,
           bound: float = _INF):
    """Cheapest route from src to dst. A* over the squeezed track.

    Every dot costs `step`. A dot of track costs `penalty` more unless one
    of the railroads on it is in the bit mask `free`. The number of moves
    left to dst is an exact lower bound on the cost, which keeps the search
    to a corridor around the answer.

    `blocked` is a set of (dot, dot, railroad), in both directions, that may
    not be ridden. `bound` is the cost of a route already in hand: anything
    that cannot beat it is dropped, and None comes back if nothing does.

    Returns (dots, runs) as Tables.shortest does, or None.
    """
    adj = t.adj
    # Kept dots use the squeezed edges, which jump straight to the next kept
    # dot. A start in the middle of a stretch walks out of it dot by dot.
    kadj = t.kadj if t.kadj[dst] is not None else adj
    to_dst = t.hops[dst]
    touched = {a for a, _, _ in blocked} if blocked else None
    dear = step + penalty
    best = [_INF] * len(adj)
    prev: list = [None] * len(adj)
    best[src] = 0
    # Entries are (cost so far + least possible cost to go, moves to go, cost
    # so far, dot). On equal totals the dot nearest dst goes first, so open
    # country is crossed in a straight line instead of being flooded.
    heap = [(to_dst[src] * step, to_dst[src], 0, src)]
    pop, push = heapq.heappop, heapq.heappush
    while heap:
        _, _, g, n = pop(heap)
        if n == dst:
            break
        if g > best[n]:
            continue
        if touched is None:
            # The usual case, with nothing closed: kept as tight as possible.
            for edge in (kadj[n] or adj[n]):
                m = edge[0]
                ng = g + edge[2] * (step if edge[4] & free else dear)
                if ng < best[m]:
                    left = to_dst[m]
                    total = ng + left * step
                    if total < bound:
                        best[m] = ng
                        prev[m] = (n, edge, edge[1])
                        push(heap, (total, left, ng, m))
            continue
        for edge in (kadj[n] or adj[n]):
            m = edge[0]
            choice = edge[1]
            mask = edge[4]
            if n in touched or m in touched:
                # Track ridden earlier this trip is closed, checked at both
                # ends of the stretch.
                inner = edge[3]
                a, b = (inner[0], inner[-1]) if inner else (m, n)
                choice = tuple([r for r in choice if (n, a, r) not in blocked and (b, m, r) not in blocked])
                if not choice:
                    continue
                mask = _bits(choice)
            ng = g + edge[2] * (step if mask & free else dear)
            if ng < best[m]:
                left = to_dst[m]
                total = ng + left * step
                if total < bound:
                    best[m] = ng
                    prev[m] = (n, edge, choice)
                    push(heap, (total, left, ng, m))
    else:
        return None

    dots, runs, n = [dst], [], dst
    while n != src:
        n, edge, choice = prev[n]
        dots.extend(reversed(edge[3]))
        dots.append(n)
        runs.append((choice, edge[2]))
    dots.reverse()
    runs.reverse()
    return tuple(dots), tuple(runs)
