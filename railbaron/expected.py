"""Expected cost of a trip, worked out exactly for the dice.

The question: standing here, heading there, with the railroads owned as they
are now, what will the rest of the trip cost on average if every roll is
played as well as possible? Cost is fees plus a price for each turn taken.

It is answered by working backwards from the destination. The destination
costs nothing. Anywhere else costs one turn, plus the average over every
roll of the cheapest way to play that roll. Within a roll the pawn moves a
step at a time, and what matters is where it is, which way it is facing (it
may not turn straight round), how many steps are left, and who has already
been paid this turn (a second ride on the same owner's track is free). The
sums are repeated until the answers stop changing.

Four simplifications, all on the side of the sums rather than the rules:

  - Track ridden on earlier turns of the trip is treated as open. The player
    choosing a move checks the real rule; see ExpectedCostPolicy.
  - A bonus roll is treated as part of one longer roll, as if the player knew
    the bonus die when making the first move.
  - The $1,000 bank fee is charged every turn. In the game a turn spent
    wholly on opponents' track escapes it, which is rare and small.
  - Only the part of the map a trip could sensibly use is worked through:
    every dot on a way there at most `slack` dots longer than the shortest.
    Beyond it a gloomy estimate stands in.

The track is handled as "arcs": a piece of track with a direction. The inner
loop is compiled by numba when it is installed, which makes it roughly a
hundred times faster. Without numba the answers are the same, only slow.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:             # pragma: no cover - the slow road
    def njit(*args, **kwargs):
        return (lambda f: f) if not (args and callable(args[0])) else args[0]

MAX_ROLL = 18           # two dice and a bonus die
BIG = 1e7               # stands for "cannot be done"
BANK_FEE = 1000


class TurnGraph:
    """The track as arcs, and every step from one arc to the next.

    An arc is (from dot, to dot). There is also one "standing start" arc per
    dot, for a pawn that has not moved yet this trip and may leave in any
    direction.
    """

    def __init__(self, tables):
        adj = tables.adj
        n = len(adj)
        ends = [(u, edge[0]) for u in range(n) for edge in adj[u]] + [(-1, d) for d in range(n)]
        number = {arc: i for i, arc in enumerate(ends)}
        self.count = len(ends)
        self.railroads = tables.rr_count
        self.arc_id = {arc: i for arc, i in number.items() if arc[0] >= 0}
        self.start = [number[(-1, d)] for d in range(n)]
        self.head = np.array([v for _, v in ends])
        self.tail = np.array([u if u >= 0 else v for u, v in ends])
        # steps[arc] = [(next arc, railroad, dot arrived at), ...]
        self.steps = [[(number[(v, edge[0])], r, edge[0]) for edge in adj[v] if edge[0] != u for r in edge[1]]
                      for u, v in ends]
        self.offsets = np.cumsum([0] + [len(out) for out in self.steps])
        self.t_to = np.array([step[0] for out in self.steps for step in out])
        self.t_rr = np.array([step[1] for out in self.steps for step in out])
        self.hops = np.array(tables.hops)
        # on_line[r]: the dots railroad r runs through
        self.on_line = [np.array([d for d in range(n) if r in tables.node_rrs[d]]) for r in range(tables.rr_count)]
        self._weights: dict = {}
        self._turns: dict = {}

    def weights(self, express: bool, superchief: bool) -> np.ndarray:
        """weights[n]: chance that a turn moves n dots in all, bonus roll included."""
        key = (express, superchief)
        w = self._weights.get(key)
        if w is None:
            w = np.zeros(MAX_ROLL + 1)
            for a in range(1, 7):
                for b in range(1, 7):
                    if superchief or (a == b and (express or a == 6)):
                        for die in range(1, 7):
                            w[a + b + die] += 1 / 216
                    else:
                        w[a + b] += 1 / 36
            self._weights[key] = w
        return w

    def turns(self, express: bool, superchief: bool) -> np.ndarray:
        """turns[h]: average number of turns to cover h dots."""
        key = (express, superchief)
        found = self._turns.get(key)
        if found is None:
            w = self.weights(express, superchief)
            top = int(self.hops.max()) + 1
            found = np.zeros(top + 1)
            for h in range(1, top + 1):
                found[h] = 1 + sum(w[n] * found[max(0, h - n)] for n in range(2, MAX_ROLL + 1))
            self._turns[key] = found
        return found


def turn_graph(tables) -> TurnGraph:
    graph = getattr(tables, "_turn_graph", None)
    if graph is None:
        graph = tables._turn_graph = TurnGraph(tables)
    return graph


@njit(cache=True)
def _solve(offsets, t_to, t_rr, head, tail, from_origin, to_goal, limit, goal, classes, fees, weights, turns,
           turn_value, warm, use_warm, tolerance, sweeps):
    count = head.size
    # ---- the part of the map worth working through
    column = np.empty(count, np.int64)
    m = 0
    for a in range(count):
        h, t = head[a], tail[a]
        if from_origin[h] + to_goal[h] <= limit and from_origin[t] + to_goal[t] <= limit:
            column[a] = m
            m += 1
        else:
            column[a] = -(to_goal[h] + 1)       # outside: remembered by its distance to go
    arcs = np.empty(m, np.int64)
    for a in range(count):
        if column[a] >= 0:
            arcs[column[a]] = a

    # ---- who can be paid here: usually fewer than everyone
    kind = np.full(fees.size, -1)       # class 0, the bank, needs no bit: its fee is in every turn
    kinds = 0
    used = np.zeros(classes.size, np.bool_)
    steps = 0
    for i in range(m):
        a = arcs[i]
        for j in range(offsets[a], offsets[a + 1]):
            steps += 1
            r = t_rr[j]
            used[r] = True
            if classes[r] and kind[classes[r]] < 0:
                kind[classes[r]] = kinds
                kinds += 1
    states = 1 << kinds

    first = np.empty(m + 1, np.int64)
    s_to = np.empty(steps, np.int64)
    s_bit = np.empty(steps, np.int64)
    s_fee = np.empty(steps, np.float64)
    at = 0
    for i in range(m):
        first[i] = at
        a = arcs[i]
        for j in range(offsets[a], offsets[a + 1]):
            s_to[at] = column[t_to[j]]
            c = classes[t_rr[j]]
            s_bit[at] = 1 << kind[c] if c else 0
            s_fee[at] = fees[c] if c else 0.0
            at += 1
    first[m] = at

    last = 0
    for n in range(weights.size):
        if weights[n] > 0:
            last = n
    worst = fees.max()
    turn_value += BANK_FEE
    outside = (turn_value + worst) * turns
    start = warm.copy() if use_warm else turn_value * turns[to_goal[head]]
    value = start[arcs]
    home = head[arcs] == goal
    table = np.empty((last + 1, m, states))
    for _ in range(sweeps):
        for i in range(m):
            table[0, i, :] = value[i]
        for n in range(1, last + 1):
            before, now = table[n - 1], table[n]
            for i in range(m):
                if home[i]:
                    now[i, :] = 0.0
                    continue
                lo, hi = first[i], first[i + 1]
                for paid in range(states):
                    best = BIG
                    for j in range(lo, hi):
                        to = s_to[j]
                        bit = s_bit[j]
                        if paid & bit:
                            v = before[to, paid] if to >= 0 else outside[-to - 1]
                        else:
                            v = s_fee[j] + (before[to, paid | bit] if to >= 0 else outside[-to - 1])
                        if v < best:
                            best = v
                    now[i, paid] = best
        change = 0.0
        for i in range(m):
            if home[i]:
                fresh = 0.0
            else:
                fresh = turn_value
                for n in range(1, last + 1):
                    if weights[n] > 0:
                        fresh += weights[n] * table[n, i, 0]
                if fresh > BIG:
                    fresh = BIG
            change = max(change, abs(fresh - value[i]))
            value[i] = fresh
        if change < tolerance:
            break
    start[arcs] = value
    return start, table, column, kind, used, outside


def trip_costs(tables, goal: int, origin: int, classes, fees, express: bool, superchief: bool,
               turn_value: float, warm=None, slack: int = 10, tolerance: float = 50.0, sweeps: int = 40):
    """Expected cost of the rest of a trip to `goal`, from anywhere near the way there from `origin`.

    classes[r]  who is paid for riding railroad r: 0 for the bank (the
                player's own track counts as the bank's), 1 and up for each
                opponent.
    fees[c]     what one turn on an opponent's track costs. fees[0] is not used.
    express, superchief   the player's train, which sets how far a turn goes.
    turn_value  dollars charged for each turn the trip takes.
    warm        `start` from an earlier, similar question, to begin from.
    slack       how many dots longer than the shortest a way there may be.

    Returns (start, steps, column, kind, railroads, outside):

    start[arc]              cost from the start of a turn, having arrived by `arc`
    steps[k][column, paid]  cost with k dots of this roll still to move
    column[arc]             where an arc is in `steps`. Negative for an arc
                            outside the part worked through: -(h + 1), where
                            `outside[h]` is the estimate that stands in.
    kind[c]                 the bit of `paid` that says opponent c has been paid
                            this turn, or -1 if they own nothing nearby
    railroads[r]            True if the answer depends on who owns railroad r
    """
    g = turn_graph(tables)
    hops = g.hops
    empty = np.empty(0)
    return _solve(g.offsets, g.t_to, g.t_rr, g.head, g.tail, hops[origin], hops[goal],
                  int(hops[origin][goal]) + slack, goal, np.asarray(classes, dtype=np.int64),
                  np.asarray(fees, dtype=np.float64), g.weights(express, superchief), g.turns(express, superchief),
                  float(turn_value), empty if warm is None else warm, warm is not None, tolerance, sweeps)
