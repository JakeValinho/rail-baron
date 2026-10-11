"""The game engine: 3 to 6 players, played to the rulebook.

    python -m railbaron.game                      watch one game in the browser
    python -m railbaron.game --games 2000         simulate 2,000 games, print a summary
    python -m railbaron.game --choices            print the region-choice table

One engine serves both uses. Run flat-out it plays about 50 games a second
per processor. Run with record=True it also writes down every roll, move, purchase
and payment, and the map page replays that recording. The page applies no
rules itself, so a rule changed here changes in both places.

This file is the rules. Every decision a player makes is in policy.py.

A turn
------
1. A player with no destination may declare (if they hold the winning
   cash), then rolls a destination. Rolling the city they are in loses the
   turn.
2. They roll the two white dice and move along their chosen route,
   stopping at the destination.
3. On arrival they collect the payoff and may buy one thing.
4. Boxcars, or doubles with an Express, or any roll with a Superchief,
   earns a bonus roll of one die. If the trip has just ended, the bonus
   roll starts the next one.
   A player who moves onto or through the dot of a declared player
   catches them: the rover play. The declared player pays $50,000, is no
   longer declared, and must go to their alternate destination.
5. They pay for the track ridden this turn: $1,000 to the bank for any of
   their own or the bank's lines, and the going rate to each opponent whose
   line they used. A player who cannot pay must sell railroads, to the bank
   at half price or by auction. With nothing left to sell they are out.

Not in the engine, or simplified
--------------------------------
* A declared player who reaches home skips any bonus roll that turn.
* The optional rules. A Superchief is always $40,000.
* Seat 1 always goes first; there is no roll for it.
* If every player but one goes out, the last one left wins.
* A bankrupt player pays what cash they have, opponents first in seat order.
"""

from __future__ import annotations

import argparse
import random
import time
from typing import Optional, Sequence, Union

from .destinations import REGIONS, Destinations
from .policy import BANK, BANK_FEE, EXPRESS, SUPERCHIEF, Policy
from .tables import Tables

STARTING_CASH = 20_000
TARGET_CASH = 200_000
EXPRESS_PRICE = 4_000
SUPERCHIEF_PRICE = 40_000
FEE = 5_000                 # paid to an opponent for using their track
FEE_ALL_SOLD = 10_000       # ... once every railroad has been bought
ROVER_PRIZE = 50_000        # paid by a declared player to the first pawn that catches them
MAX_ROUNDS = 2_000

MOVE = tuple(a + b for a in range(1, 7) for b in range(1, 7))   # two dice, by roll number 0..35
DEFAULT_POLICY = Policy()


class Player:
    __slots__ = ("index", "name", "policy", "home", "city", "destination", "trip_from", "goal",
                 "dots", "rrs", "step", "cash", "trips", "express", "superchief", "lines", "access",
                 "own_mask", "est", "declared", "at_home", "trail", "replan", "out")

    def __init__(self, index: int, name: str, policy: Policy, cash: int):
        self.index = index
        self.name = name
        self.policy = policy
        self.home = -1              # city number
        self.city = -1              # city the current trip started from, or the last one reached
        self.destination = -1       # city the trip pays off at, or -1 between trips
        self.trip_from = -1
        self.goal = -1              # dot being travelled to: the destination, or home when declared
        self.dots: tuple = ()       # the planned route, dot by dot
        self.rrs: tuple = ()        # railroad ridden on each step of it
        self.step = 0               # steps of the route already taken
        self.cash = cash
        self.trips = 0
        self.express = False
        self.superchief = False
        self.lines: list = []       # railroads owned
        self.access = 0             # cities those railroads reach, as a bit mask
        self.own_mask = 0           # the railroads themselves, as a bit mask
        self.est: Optional[dict] = None     # "established": railroad -> old fee still being paid
        self.declared = False
        self.at_home = False        # declared and standing in the home city
        self.trail: Optional[list] = None   # track ridden this trip before a re-plan
        self.replan = False
        self.out = False

    @property
    def dot(self) -> int:
        """The dot the player is standing on."""
        return self.dots[self.step]


class Game:
    def __init__(
        self,
        players: int = 4,
        seed: Optional[int] = None,
        tables: Optional[Tables] = None,
        policies: Union[Policy, Sequence[Policy], None] = None,
        starting_cash: int = STARTING_CASH,
        target_cash: int = TARGET_CASH,
        record: bool = False,
    ):
        if not 3 <= players <= 6:
            raise ValueError("Rail Baron is for 3 to 6 players")
        self.tables = t = tables or Tables()
        self.dest = t.destinations
        self.target_cash = target_cash
        self.express_price = EXPRESS_PRICE
        self.superchief_price = SUPERCHIEF_PRICE
        self.rover_prize = ROVER_PRIZE
        self.rand = random.Random(seed).random
        if policies is None:
            policies = DEFAULT_POLICY
        if isinstance(policies, Policy):
            policies = [policies] * players
        self.players = [Player(i, f"Player {i + 1}", policies[i], starting_cash) for i in range(players)]

        self.owner = [BANK] * t.rr_count    # who owns each railroad
        self.sold_mask = 0                  # railroads owned by players, as a bit mask
        self.bank_count = t.rr_count
        self.rate = FEE                     # what an opponent's track costs
        self.seat = 0                       # whose turn it is
        self.turn = 0                       # turns taken so far, all players together
        self.winner: Optional[int] = None
        self.declared = 0                   # how many players are declared and heading home
        self.bought = []                    # (railroad, turn) for each purchase from the bank, in order
        self.bank_due = 0                   # fees run up by the player now moving
        self.due: dict = {}
        self.recording: Optional[dict] = None
        self.route_note = self.buy_note = None

        homes, taken = [], set()
        for p in self.players:
            while True:                     # no two players share a home city
                city, log = self._roll_city(None)
                if t.city_place[city] not in taken:
                    break
            taken.add(t.city_place[city])
            p.home = p.city = city
            p.dots = (t.city_node[city],)
            homes.append(log)
        if record:
            self.recording = {
                "players": [{"name": p.name, "home": p.home, "roll": log} for p, log in zip(self.players, homes)],
                "start_cash": starting_cash,
                "target_cash": target_cash,
                "turns": [],
                "winner": None,
            }

    # ------------------------------------------------------- destinations

    def _roll_city(self, p: Optional[Player]):
        """Roll region then city. Returns (city, what was rolled)."""
        t, rand = self.tables, self.rand
        k = int(rand() * 216)
        rolled = region = t.region_by_roll[k]
        if p is not None and region == self.dest.region[p.city]:
            region = p.policy.choose_region(self, p)            # PLAYER CHOICE 1
        c = int(rand() * 216)
        city = t.city_by_roll[region][c]
        return city, {"region_roll": k, "rolled": rolled, "region": region, "city_roll": c, "city": city}

    def _start_trip(self, p: Player, ev) -> bool:
        """Declare if wanted, roll a destination, plan the route.

        Returns False if the player has no trip to move on: they won on the
        spot, or rolled the city they are in and lose the turn.
        """
        t = self.tables
        declared = False
        if p.cash >= self.target_cash and p.policy.choose_declare(self, p):     # PLAYER CHOICE 4
            if t.city_place[p.city] == t.city_place[p.home]:
                self.winner = p.index
                if ev is not None:
                    ev.append({"t": "win", "how": "declared at home"})
                return False
            declared = True
        city, log = self._roll_city(p)
        lost = t.city_place[city] == t.city_place[p.city]
        if ev is not None:
            log.update(t="dest", lost=lost, declared=declared and not lost)
            ev.append(log)
        if lost:
            return False
        p.destination, p.trip_from = city, p.city
        p.declared, p.at_home, p.trail = declared, False, None
        self.declared += declared
        goal = t.city_node[p.home] if declared else t.city_node[city]
        self._plan(p, t.city_node[p.city], goal, None, ev)
        return True

    def _plan(self, p: Player, src: int, goal: int, blocked, ev) -> None:
        p.dots, p.rrs = p.policy.choose_route(self, p, src, goal, blocked)     # PLAYER CHOICE 2
        p.step, p.goal, p.replan = 0, goal, False
        if ev is not None:
            ids = self.tables.node_ids
            ev.append({"t": "route", "p": p.index, "nodes": [ids[n] for n in p.dots], "rrs": list(p.rrs),
                       "options": self.route_note})

    def _ridden(self, p: Player):
        """Track ridden so far this trip: (the list, the same as a blocked set)."""
        trail = list(p.trail or [])
        trail.extend(zip(p.dots, p.dots[1:p.step + 1], p.rrs))
        blocked = set()
        for a, b, r in trail:           # each piece of track is ridden once per trip
            blocked.add((a, b, r))
            blocked.add((b, a, r))
        return trail, blocked

    def _replan(self, p: Player, ev) -> None:
        """Choose a new route from where the player stands, mid-trip."""
        p.trail, blocked = self._ridden(p)
        self._plan(p, p.dots[p.step], p.goal, blocked, ev)

    def _rover(self, p: Player, ev) -> None:
        """Before p moves: let them turn aside to chase a declared player."""
        for q in self.players:
            if q.declared and q is not p and not p.declared and p.step < len(p.rrs):
                trail, blocked = self._ridden(p)
                route = p.policy.choose_rover(self, p, q, blocked)             # PLAYER CHOICE 7
                if route is None:
                    continue
                dots, rrs = route
                if dots[0] != p.dots[p.step] or dots[-1] != p.goal or len(rrs) != len(dots) - 1:
                    raise ValueError("a rover route must run from where the player is to their destination")
                p.trail, p.dots, p.rrs, p.step, p.replan = trail, dots, rrs, 0, False
                if ev is not None:
                    ids = self.tables.node_ids
                    ev.append({"t": "rover", "target": q.index, "nodes": [ids[n] for n in dots], "rrs": list(rrs)})
                return

    def _catch(self, p: Player, q: Player, ev) -> None:
        """p has moved onto or through declared player q: the rover play."""
        paid = min(self.rover_prize, q.cash)
        q.cash -= paid
        p.cash += paid
        if ev is not None:
            ev.append({"t": "caught", "p": q.index, "by": p.index, "pay": paid})
        self._undeclare(q, ev)

    # ----------------------------------------------------------- movement

    def _move(self, p: Player, count: int, ev, dice) -> None:
        if self.declared:
            self._rover(p, ev)
        if p.replan and p.step < len(p.rrs):
            self._replan(p, ev)
        start, last = p.step, len(p.rrs)
        to = min(start + count, last)

        # Note who is owed for the track ridden. Paid at the end of the turn.
        owner, me, rate, due, est = self.owner, p.index, self.rate, self.due, p.est
        for r in p.rrs[start:to]:
            o = owner[r]
            fee = rate
            if est is not None:
                old = est.get(r)
                if old is None:
                    est = p.est = None          # left the line: the old fee is gone
                else:
                    if len(est) > 1:
                        est = p.est = {r: old}
                    fee = old
            if o < 0 or o == me:
                self.bank_due = BANK_FEE
            elif due.get(o, 0) < fee:
                due[o] = fee

        p.step = to
        if ev is not None:
            ev.append({"t": "move", "dice": dice, "bonus": len(dice) == 1, "from": start, "to": to})
        if self.declared:
            # The rover play: only the first pawn to reach a declared player
            # collects, because being caught ends the declaration.
            passed = p.dots[start + 1:to + 1]
            for q in self.players:
                if q.declared and q is not p and q.dots[q.step] in passed:
                    self._catch(p, q, ev)
        if to == last:
            self._arrive(p, ev)

    def _arrive(self, p: Player, ev) -> None:
        t = self.tables
        if p.declared:
            p.at_home = True
            if t.city_place[p.destination] != t.city_place[p.home]:
                if ev is not None:
                    ev.append({"t": "home", "p": p.index})
                return                  # no payoff for reaching home; the win is settled after fees
        pay = self.dest.payoff[p.trip_from][p.destination]
        p.cash += pay
        p.trips += 1
        if ev is not None:
            ev.append({"t": "arrive", "p": p.index, "city": p.destination, "from": p.trip_from, "pay": pay})
        p.city, p.destination = p.destination, -1

        choice = p.policy.choose_purchase(self, p)                              # PLAYER CHOICE 3
        if choice is None:
            return
        if choice == EXPRESS:
            if p.express or p.cash < self.express_price:
                raise ValueError("cannot buy an Express")
            p.cash -= self.express_price
            p.express = True
            price = self.express_price
        elif choice == SUPERCHIEF:
            if p.superchief or p.cash < self.superchief_price:
                raise ValueError("cannot buy a Superchief")
            p.cash -= self.superchief_price
            p.superchief = True
            price = self.superchief_price
        else:
            price = t.rr_price[choice]
            if self.owner[choice] != BANK or p.cash < price:
                raise ValueError(f"cannot buy railroad {choice}")
            p.cash -= price
            self._transfer(choice, p.index)
            self.bought.append((choice, self.turn))
        if ev is not None:
            item = "express" if choice == EXPRESS else "superchief" if choice == SUPERCHIEF else "line"
            ev.append({"t": "buy", "p": p.index, "item": item, "rr": choice if choice >= 0 else None, "price": price,
                       "considered": self.buy_note if choice >= 0 else None})

    # ---------------------------------------------------------- ownership

    def _transfer(self, r: int, new: int) -> None:
        """Railroad r changes hands. `new` is a player number or BANK."""
        t, players = self.tables, self.players
        old = self.owner[r]
        self.owner[r] = new
        bit = 1 << r
        if old >= 0:
            q = players[old]
            q.lines.remove(r)
            q.own_mask &= ~bit
            q.access = 0
            for line in q.lines:
                q.access |= t.rr_mask[line]
        if new >= 0:
            players[new].lines.append(r)
            players[new].own_mask |= bit
            players[new].access |= t.rr_mask[r]
            self.sold_mask |= bit
        else:
            self.sold_mask &= ~bit
        self.bank_count += (new < 0) - (old < 0)

        for q in players:
            if q.out:
                continue
            was_free = old < 0 or old == q.index
            now_free = new < 0 or new == q.index
            if was_free and not now_free:
                # The fee for q just went up. Standing on the line, q keeps
                # paying the old $1,000 for as long as they stay on it.
                if r in t.node_rrs[q.dots[q.step]]:
                    if q.est is None:
                        q.est = {}
                    q.est.setdefault(r, BANK_FEE)
                if q.destination >= 0 and q.policy.REPLAN and r in q.rrs[q.step:]:
                    q.replan = True         # their planned route just got dearer

        if self.bank_count == 0 and self.rate == FEE:
            # Every railroad is sold: the fee doubles for the rest of the game.
            self.rate = FEE_ALL_SOLD
            for q in players:
                if q.out:
                    continue
                for line in t.node_rrs[q.dots[q.step]]:
                    if 0 <= self.owner[line] != q.index:
                        if q.est is None:
                            q.est = {}
                        q.est.setdefault(line, FEE)

    def _auction(self, seller: Player, r: int):
        """Auction railroad r. Returns (buyer, price, the bids made)."""
        n, players = len(self.players), self.players
        bidders = [players[(seller.index + k) % n] for k in range(1, n)]
        bidders = [q for q in bidders if not q.out]
        limit = {q.index: min(q.cash, q.policy.max_bid(self, q, r)) for q in bidders}    # PLAYER CHOICE 6
        minimum = self.tables.rr_price[r] // 2
        high, leader, bids, passes = 0, None, [], 0
        while bidders and passes < len(bidders):
            for q in bidders:
                if q is leader:
                    passes += 1
                elif limit[q.index] >= (ask := max(minimum, high + 500)):
                    high, leader, passes = ask, q, 0
                    bids.append([q.index, ask])
                else:
                    passes += 1
                if passes >= len(bidders):
                    break
        if leader is None:
            return BANK, minimum, bids          # nobody bid: the bank takes it at half price
        return leader.index, high, bids

    def _raise_cash(self, p: Player, owed: int, ev) -> None:
        """Sell railroads until the fees can be paid, or there are none left."""
        while p.cash < owed and p.lines:
            r, auction = p.policy.choose_sale(self, p, owed - p.cash)          # PLAYER CHOICE 5
            if r not in p.lines:
                raise ValueError(f"cannot sell railroad {r}")
            if auction:
                buyer, price, bids = self._auction(p, r)
            else:
                buyer, price, bids = BANK, self.tables.rr_price[r] // 2, []
            p.cash += price
            if buyer >= 0:
                self.players[buyer].cash -= price
            self._transfer(r, buyer)
            if ev is not None:
                ev.append({"t": "sale", "rr": r, "to": buyer, "price": price, "auction": auction, "bids": bids})
            if buyer >= 0:
                self._check_declared(self.players[buyer], ev)

    # ------------------------------------------------------ end of a turn

    def _settle(self, p: Player, ev) -> None:
        """Pay the turn's fees, then see whether the game is over."""
        due, players = self.due, self.players
        owed = self.bank_due + sum(due.values())
        if owed:
            if p.cash < owed:
                self._raise_cash(p, owed, ev)
            if ev is not None:
                ev.append({"t": "fees", "bank": self.bank_due, "to": sorted(due.items())})
            if p.cash >= owed:
                p.cash -= owed
                for o, amount in due.items():
                    players[o].cash += amount
            else:
                for o, amount in sorted(due.items()):       # bankrupt: pay out what there is
                    paid = min(amount, p.cash)
                    players[o].cash += paid
                    p.cash -= paid
                p.cash = 0
                p.out = True
                self.declared -= p.declared
                p.declared = False
                p.destination = -1
                if ev is not None:
                    ev.append({"t": "out"})
                alive = [q for q in players if not q.out]
                if len(alive) == 1 and self.winner is None:
                    self.winner = alive[0].index
                    if ev is not None:
                        ev.append({"t": "win", "how": "last player left", "p": self.winner})
        if not p.out:
            self._check_declared(p, ev)

    def _check_declared(self, p: Player, ev) -> None:
        """A declared player wins at home with the cash, or loses the right."""
        if not p.declared or self.winner is not None:
            return
        if p.cash >= self.target_cash:
            if p.at_home:
                self.winner = p.index
                if ev is not None:
                    ev.append({"t": "win", "how": "reached home", "p": p.index})
            return
        self._undeclare(p, ev)          # fell below the target

    def _undeclare(self, p: Player, ev) -> None:
        """No longer declared: carry on to the alternate destination, as
        part of the same trip."""
        p.declared = p.at_home = False
        self.declared -= 1
        if ev is not None:
            ev.append({"t": "undeclare", "p": p.index})
        if p.destination < 0:
            return                      # home was the alternate, and the trip is already paid
        p.goal = self.tables.city_node[p.destination]
        if p.dots[p.step] == p.goal:
            self._arrive(p, ev)
        else:
            self._replan(p, ev)

    def play_turn(self) -> None:
        """One player's whole turn."""
        p = self.players[self.seat]
        ev = [] if self.recording is not None else None
        self.bank_due = 0
        self.due = {}

        if p.destination >= 0 or self._start_trip(p, ev):
            m = int(self.rand() * 36)
            d1, d2 = m // 6, m % 6
            bonus = p.superchief or (d1 == d2 and (p.express or d1 == 5))
            self._move(p, d1 + d2 + 2, ev, [d1 + 1, d2 + 1])
            if bonus and not p.at_home and (p.destination >= 0 or self._start_trip(p, ev)):
                die = int(self.rand() * 6) + 1
                self._move(p, die, ev, [die])
        self._settle(p, ev)

        if ev is not None:
            self.recording["turns"].append({"p": p.index, "ev": ev, "cash": [q.cash for q in self.players]})
            self.recording["winner"] = self.winner
        self.turn += 1
        n = len(self.players)
        for k in range(1, n + 1):           # on to the next player still in the game
            nxt = (self.seat + k) % n
            if not self.players[nxt].out:
                self.seat = nxt
                break

    def play(self, max_rounds: int = MAX_ROUNDS) -> "Game":
        """Play until someone wins or `max_rounds` full rounds have passed."""
        limit = max_rounds * len(self.players)
        while self.winner is None and self.turn < limit:
            self.play_turn()
        return self

    @property
    def rounds(self) -> int:
        return -(-self.turn // len(self.players))


# ------------------------------------------------------------ many games


_WORKER_TABLES: Optional[Tables] = None


def _play_batch(job) -> list:
    """Play one batch of games and return its totals. Runs in a worker."""
    global _WORKER_TABLES
    players, seeds, options, tables = job
    if tables is None:
        if _WORKER_TABLES is None:
            _WORKER_TABLES = Tables()
        tables = _WORKER_TABLES
    wins = [0] * players
    rounds = trips = lines = sold = out = unfinished = superchiefs = 0
    count = tables.rr_count
    times, order, when = [0] * count, [0] * count, [0] * count     # first purchase of each railroad
    for s in seeds:
        g = Game(players, s, tables, **options).play()
        rounds += g.rounds
        sold += tables.rr_count - g.bank_count
        out += sum(p.out for p in g.players)
        superchiefs += sum(p.superchief for p in g.players)
        seen = 0
        for r, turn in g.bought:
            if not seen >> r & 1:                   # a line sold back and bought again counts once
                times[r] += 1
                order[r] += seen.bit_count() + 1
                when[r] += turn // players + 1
                seen |= 1 << r
        if g.winner is None:
            unfinished += 1
        else:
            w = g.players[g.winner]
            wins[g.winner] += 1
            trips += w.trips
            lines += len(w.lines)
    return [rounds, trips, lines, sold, out, unfinished, superchiefs] + wins + times + order + when


def simulate(games: int, players: int = 4, seed: Optional[int] = None,
             tables: Optional[Tables] = None, workers: int = 1, **options) -> dict:
    """Play `games` games without recording them and summarise the results.

    `workers` spreads the games over that many processes (0 means one per
    processor). The result is the same whatever the number of workers.
    Other options, such as policies=..., are passed on to Game.
    """
    tables = tables or Tables()
    rng = random.Random(seed)
    seeds = [rng.getrandbits(48) for _ in range(games)]
    started = time.perf_counter()
    if workers == 1 or games < 50:
        totals = _play_batch((players, seeds, options, tables))
    else:
        import multiprocessing

        workers = workers or multiprocessing.cpu_count()
        size = max(1, games // (workers * 4))
        jobs = [(players, seeds[i:i + size], options, None) for i in range(0, games, size)]
        with multiprocessing.Pool(workers) as pool:
            totals = [sum(column) for column in zip(*pool.imap_unordered(_play_batch, jobs))]
    seconds = time.perf_counter() - started
    rounds, trips, lines, sold, out, unfinished, superchiefs = totals[:7]
    finished = max(1, games - unfinished)
    return {
        "games": games,
        "players": players,
        "seconds": seconds,
        "games_per_second": games / seconds,
        "average_rounds": rounds / games,
        "win_share_by_seat": [w / games for w in totals[7:7 + players]],
        "average_winner_trips": trips / finished,
        "average_winner_railroads": lines / finished,
        "average_railroads_sold": sold / games,
        "average_superchiefs": superchiefs / games,
        "bankruptcies_per_game": out / games,
        "unfinished": unfinished,
        "purchases": _purchases(tables, games, totals[7 + players:]),
    }


def _purchases(tables: Tables, games: int, columns: list) -> list[dict]:
    """Per railroad, earliest bought first: how often, in what order and how early it is bought."""
    n = tables.rr_count
    times, order, when = columns[:n], columns[n:2 * n], columns[2 * n:]
    rows = [{"railroad": tables.rr_ids[r], "name": tables.rr_name[r], "price": tables.rr_price[r],
             "bought_share": times[r] / games,
             "average_order": order[r] / times[r] if times[r] else None,
             "average_round": when[r] / times[r] if times[r] else None}
            for r in range(n)]
    return sorted(rows, key=lambda row: (row["average_order"] is None, row["average_order"]))


def purchase_report(summary: dict) -> str:
    """The purchase-order table for a simulate() result."""
    lines = [f"  {'railroad':10}{'price':>9}{'bought':>9}{'order':>8}{'round':>8}"]
    for row in summary["purchases"]:
        if row["average_order"] is None:
            lines.append(f"  {row['railroad']:10}{'$' + format(row['price'], ','):>9}{'never':>9}")
        else:
            lines.append(f"  {row['railroad']:10}{'$' + format(row['price'], ','):>9}{row['bought_share']:>9.1%}"
                         f"{row['average_order']:>8.1f}{row['average_round']:>8.1f}")
    return "\n".join(lines)


def record_games(player_counts: Sequence[int] = (3, 4, 5, 6), seed: Optional[int] = None,
                 tables: Optional[Tables] = None) -> list[dict]:
    """Play one recorded game per player count, for the map page to replay."""
    tables = tables or Tables()
    seeds = random.Random(seed)
    return [Game(n, seeds.getrandbits(48), tables, record=True).play().recording for n in player_counts]


def region_choice_report(destinations: Destinations) -> str:
    """The region-choice table: region taken from each city, and why."""
    d = destinations
    lines = [f"{'from':18}{'takes':>6}   average payoff by region ($1000s)",
             f"{'':24}   " + "".join(f"{r:>7}" for r in REGIONS)]
    for c in sorted(range(d.count), key=lambda c: (d.region[c], d.names[c])):
        values = [d.expected_payoff_in_region(c, r) / 1000 for r in range(len(REGIONS))]
        lines.append(f"{d.names[c]:18}{REGIONS[d.richest_region[c]]:>6}   " + "".join(f"{v:7.1f}" for v in values))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play or simulate Rail Baron.")
    parser.add_argument("--players", type=int, default=4, help="3 to 6 (default 4)")
    parser.add_argument("--games", type=int, default=0, help="simulate this many games instead of watching one")
    parser.add_argument("--seed", type=int, default=None, help="repeat the same dice")
    parser.add_argument("--workers", type=int, default=1,
                        help="processes to simulate with; 0 uses every processor (default 1)")
    parser.add_argument("--purchases", action="store_true",
                        help="with --games: also print the order in which railroads are bought")
    parser.add_argument("--choices", action="store_true", help="print the region-choice table and stop")
    parser.add_argument("--no-open", action="store_true", help="write map.html without opening it")
    args = parser.parse_args()

    tables = Tables()
    if args.choices:
        print(region_choice_report(tables.destinations))
        return

    if args.games:
        s = simulate(args.games, args.players, args.seed, tables, workers=args.workers)
        print(f"{s['games']:,} games with {s['players']} players in {s['seconds']:.2f} s "
              f"({s['games_per_second']:,.0f} games a second)")
        print(f"  average length       {s['average_rounds']:.1f} rounds")
        print(f"  winner's trips       {s['average_winner_trips']:.1f}")
        print(f"  winner's railroads   {s['average_winner_railroads']:.1f}")
        print(f"  railroads sold       {s['average_railroads_sold']:.1f} of {tables.rr_count}")
        print(f"  Superchiefs bought   {s['average_superchiefs']:.2f} a game")
        print(f"  bankruptcies         {s['bankruptcies_per_game']:.3f} a game")
        print(f"  unfinished games     {s['unfinished']}")
        print("  wins by seat         " + "  ".join(f"{i + 1}: {w:.1%}" for i, w in enumerate(s["win_share_by_seat"])))
        if args.purchases:
            print("\nRailroads, earliest bought first. Order is 1 for the first railroad bought in a game;")
            print("round is the round of play it was bought in. Both are averages over games where it was bought.")
            print(purchase_report(s))
        return

    import webbrowser

    from .render import render

    # The requested player count comes first, so the page opens on it.
    counts = [args.players] + [n for n in (3, 4, 5, 6) if n != args.players]
    out = render(board=tables.board, games=record_games(counts, args.seed, tables), open_tab="paneGame")
    print(f"wrote {out}")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
