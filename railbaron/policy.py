"""Every decision a player makes. Everything else in the game is dice.

A player can influence the game in exactly seven ways. Each one is a method
on Policy, marked PLAYER CHOICE. To try a different way of playing, subclass
Policy, override the method, and hand it to Game(policies=...).

    1. choose_region     which region to take after rolling your own region
    2. choose_route      which way to travel, and which railroad on shared track
    3. choose_purchase   what to buy after a payoff, if anything
    4. choose_declare    whether to head home once you have the winning cash, or wait
    5. choose_sale       which railroad to give up when you cannot pay, and how
    6. max_bid           how high to go when someone else's railroad is auctioned
    7. choose_rover      whether to turn aside and chase a declared player

The defaults below are simple, fast rules of thumb, not strong play. The
numbers at the top of the class are the knobs they turn on.
"""

from __future__ import annotations

from typing import Optional

from .tables import search

BANK = -1          # owner number for a railroad nobody has bought
EXPRESS = -2       # what choose_purchase returns to buy an Express
SUPERCHIEF = -3    # ... or a Superchief. 0 and up is a railroad number.
BANK_FEE = 1000


class Policy:
    # ---- route choice
    STEP_VALUE = 500        # dollars a player would pay to save one dot of travel
    MOVE = 7                # dots expected per turn, used to split a route into turns
    MOVE_SUPERCHIEF = 10
    AVOID = 3.0             # how hard the detour search steers off opponent track
    REPLAN = True           # look for a new route mid-trip when the planned one gets dearer

    # ---- what a railroad is worth
    FEE_OUTLOOK = 15000     # dollars a railroad earns or saves each time it matters (set by tournament)
    TRIP_INCOME = 15000     # rough net income per trip, used to guess how long the game has left
    HORIZON_MIN, HORIZON_MAX = 2, 14    # trips each player still has to make
    COVER_WEIGHT = 0.5      # reaching a city on your own track
    LOCK_WEIGHT = 1.0       # a city an opponent can then only reach by paying
    TOLL_WEIGHT = 0.5       # track that shortest routes are forced over
    MIN_RETURN = 1.0        # buy only if value is at least this many times the price

    # ---- declaring
    DECLARE_PATIENCE = 1.0  # declare if home is no further than this many times a typical trip home
    DECLARE_REACH = 12      # an opponent within this many dots of the way home may catch the player
    DECLARE_THREAT = 1.0    # how much each such opponent shortens the trip the player will risk
    DECLARE_URGENT = 0.75   # stop waiting once an opponent has this share of the winning cash

    # ---- the rover play
    STOP_VALUE = 50000      # what stopping someone else's win is worth, on top of the prize

    # ---- cash kept back
    RESERVE_TURNS = 3       # turns of bank fees to keep in hand

    # ------------------------------------------------------------------
    # 1. PLAYER CHOICE: region, after rolling the region you are in.
    #
    # Takes the region with the highest average payoff from the current
    # city. The calculation is Destinations.richest_region: for each region,
    # the payoff to every city in it, weighted by the city dice. It ignores
    # how far away the region is and who owns the track on the way.
    # ------------------------------------------------------------------
    def choose_region(self, game, p) -> int:
        return game.dest.richest_region[p.city]

    # ------------------------------------------------------------------
    # 2. PLAYER CHOICE: route, and which railroad to ride on shared track.
    #
    # The route engine. It weighs two routes in dollars and takes the
    # lighter one:
    #
    #     shortest   fewest dots, whoever owns the track
    #     detour     found by one search that treats each dot of opponent
    #                track as costing AVOID x fee / MOVE on top of the dot
    #
    #     weight = dots x STEP_VALUE + the fees the route is expected to cost
    #
    # Fees are estimated by cutting the route into turns of MOVE dots and
    # charging each turn as the rules would. If the shortest route touches
    # no opponent track it is taken at once with no search, which is the
    # usual case early in the game.
    #
    # `blocked` is track already ridden this trip, when re-planning part
    # way through (see REPLAN).
    #
    # Returns (dots, railroads): the dots in order, and the railroad ridden
    # on each step.
    # ------------------------------------------------------------------
    def choose_route(self, game, p, src: int, dst: int, blocked: Optional[set] = None):
        t = game.tables
        theirs = game.sold_mask & ~p.own_mask       # railroads owned by opponents
        explain = game.recording is not None
        candidates = []
        if blocked is None:
            first = t.shortest(src, dst)
            if not first[2] & theirs:
                # The shortest route is all the player's own or the bank's track.
                if explain:
                    game.route_note = [{"dots": len(first[3]), "fees": None, "picked": True}]
                return first[0], first[3]
            candidates.append(first)

        rate = game.rate
        move = self.MOVE_SUPERCHIEF if p.superchief else self.MOVE
        step = self.STEP_VALUE
        free = ~theirs
        if p.est:
            for r, old in p.est.items():        # track still ridden at the old bank fee
                if old == BANK_FEE:
                    free |= 1 << r
        penalty = int(rate * self.AVOID) // move
        if candidates:
            # What the shortest route costs the search. Only better is wanted.
            bound = 0
            for choice, count in first[1]:
                for r in choice:
                    if free >> r & 1:
                        bound += count * step
                        break
                else:
                    bound += count * (step + penalty)
            found = search(t, src, dst, free, step, penalty, None, bound)
        else:
            found = search(t, src, dst, free, step, penalty, blocked)
        if found is not None:
            candidates.append(found)
        if not candidates:
            # Only an interrupted homeward trip permits unavoidable reuse.
            # The engine determines which sections may be reused, and how many.
            if p.reusable:
                return game._reuse_route(p, src, dst, blocked or set())
            raise ValueError("no legal route to the destination")

        owner, me = game.owner, p.index
        best = best_weight = best_picks = None
        notes = []
        for route in candidates:
            length = len(route[0]) - 1
            fees, picks = weigh(route[1], owner, me, rate, move)
            weight = length * step + fees
            if explain:
                notes.append({"dots": length, "fees": fees, "picked": False})
            if best_weight is None or weight < best_weight:
                best, best_weight, best_picks = route, weight, picks
        if explain:
            notes[candidates.index(best)]["picked"] = True
            game.route_note = notes
        rrs = []
        for r, (_, count) in zip(best_picks, best[1]):
            rrs += [r] * count
        return best[0], tuple(rrs)

    # ------------------------------------------------------------------
    # 3. PLAYER CHOICE: what to buy after a payoff. One item, or nothing.
    #
    # Express: bought as soon as it can be paid for.
    # Superchief: bought only once the bank has no railroads left.
    # Railroad: the one whose value most exceeds its price, see line_value(),
    #   provided the player keeps reserve() in hand to finish the next trip.
    #
    # Returns a railroad number, EXPRESS, SUPERCHIEF, or None.
    # ------------------------------------------------------------------
    def choose_purchase(self, game, p):
        cash = p.cash
        if not p.express and not p.superchief and cash >= game.express_price:
            return EXPRESS
        spare = cash - self.reserve(game, p)
        if not game.bank_count:
            return SUPERCHIEF if not p.superchief and spare >= game.superchief_price else None
        prices = game.tables.rr_price
        outlook = None
        best, best_net, notes = None, 0.0, []
        for r, o in enumerate(game.owner):
            if o >= 0 or prices[r] > spare:
                continue
            if outlook is None:
                outlook = self.outlook(game, p)
            value = self.line_value(game, p, r, outlook)
            if game.recording is not None:
                notes.append((value - prices[r], r, value))
            net = value - prices[r] * self.MIN_RETURN
            if net > best_net:
                best, best_net = r, net
        if game.recording is not None:
            game.buy_note = [{"rr": r, "value": round(v)} for _, r, v in sorted(notes, reverse=True)[:3]]
        return best

    # ------------------------------------------------------------------
    # 4. PLAYER CHOICE: declare and head home, once cash reaches the target.
    #
    # Declining means taking an ordinary trip, with its payoff, and being
    # asked again where that trip ends. Four things are weighed:
    #
    #   distance   Turns to get home from here, against the turns it takes
    #              from a typical destination. Further than typical: another
    #              trip will probably end closer, so wait.
    #   train      A Superchief covers the distance in fewer turns.
    #   threats    Each opponent within DECLARE_REACH dots of the way home
    #              shortens the trip the player is willing to risk.
    #   urgency    If an opponent is declared, or has DECLARE_URGENT of the
    #              winning cash, there is no time to wait: declare.
    #
    # It ignores how much cash the player has above the target, and so the
    # chance of fees pulling them back under it on the way.
    # ------------------------------------------------------------------
    def choose_declare(self, game, p) -> bool:
        t = game.tables
        here, home = t.city_node[p.city], t.city_node[p.home]
        if here == home:
            return True                         # wins on the spot
        rivals = [q for q in game.players if q is not p and not q.out]
        pressing = game.target_cash * self.DECLARE_URGENT
        for q in rivals:
            if q.declared or q.cash >= pressing:
                return True
        hops = t.hops
        threats = 0
        if self.DECLARE_THREAT:
            way = t.shortest(here, home)[0]
            reach = self.DECLARE_REACH
            for q in rivals:
                near = hops[q.dots[q.step] if q.dots else t.city_node[q.city]]
                for d in way:
                    if near[d] <= reach:
                        threats += 1
                        break
        turns = hops[here][home] / (self.MOVE_SUPERCHIEF if p.superchief else self.MOVE)
        typical = t.typical_hops(home) / self.MOVE
        return turns * (1 + self.DECLARE_THREAT * threats) <= self.DECLARE_PATIENCE * typical

    # ------------------------------------------------------------------
    # 5. PLAYER CHOICE: forced sale. Which railroad, and bank or auction.
    #
    # Gives up the railroad it values least, and always auctions it, since
    # an auction never brings in less than the bank's half price. The risk
    # it ignores: an auction can hand the line to an opponent.
    #
    # Returns (railroad, auction). `short` is the cash still to be found.
    # ------------------------------------------------------------------
    def choose_sale(self, game, p, short: int):
        outlook = self.outlook(game, p)
        return min(p.lines, key=lambda r: self.line_value(game, p, r, outlook)), True

    # ------------------------------------------------------------------
    # 6. PLAYER CHOICE: the most to bid for railroad r in an auction.
    #
    # Bids up to what line_value() says the line is worth, never dipping
    # into reserve(). A declared player does not bid. Return 0 to pass.
    # ------------------------------------------------------------------
    def max_bid(self, game, p, r: int) -> int:
        if p.declared:
            return 0
        value = self.line_value(game, p, r, self.outlook(game, p))
        return max(0, int(min(value, p.cash - self.reserve(game, p))))

    # ------------------------------------------------------------------
    # 7. PLAYER CHOICE: the rover play. Chase a declared player or not.
    #
    # Asked before each move while player q is declared. Catching q means
    # moving onto or through the dot q stands on, so only this turn's roll
    # matters: next turn q will be somewhere else.
    #
    # Chases when the chance of rolling far enough, times the prize plus
    # STOP_VALUE, beats the cost of the detour in travel time. The detour is
    # the extra dots of going by way of q, valued at STEP_VALUE each. Fees
    # on the detour are ignored.
    #
    # `blocked` is the track already ridden this trip. Returns a route
    # (dots, railroads) from where p stands, through q's dot, on to p's own
    # destination, never using a piece of track twice. Returns None to
    # carry on as planned.
    # ------------------------------------------------------------------
    def choose_rover(self, game, p, q, blocked: set):
        t = game.tables
        here, there = p.dots[p.step], q.dots[q.step]
        away = t.hops[here][there]
        if away == 0 or away > 12:
            return None
        ahead = p.dots[p.step + 1:p.step + 13]
        if there in ahead:
            return None                     # already on the way
        chance = REACH[away]
        if p.superchief:
            chance = min(1.0, chance + 0.25)
        detour = away + t.hops[there][p.goal] - (len(p.rrs) - p.step)
        if chance * (game.rover_prize + self.STOP_VALUE) <= detour * self.STEP_VALUE:
            return None

        theirs = game.sold_mask & ~p.own_mask
        penalty = int(game.rate * self.AVOID) // self.MOVE
        out = search(t, here, there, ~theirs, self.STEP_VALUE, penalty, blocked)
        if out is None or len(out[0]) - 1 > 12:
            return None
        if p.goal in out[0][:-1]:
            return None  # arrival ends movement before the pawn can reach q
        owner, me = game.owner, p.index
        first = weigh(out[1], owner, me, game.rate, self.MOVE)[1]
        rrs = []
        for r, (_, count) in zip(first, out[1]):
            rrs += [r] * count
        used = set(blocked)
        for a, b, r in zip(out[0], out[0][1:], rrs):
            used.add((a, b, r))
            used.add((b, a, r))
        if there == p.goal:
            return out[0], tuple(rrs)
        back = search(t, there, p.goal, ~theirs, self.STEP_VALUE, penalty, used)
        if back is None:
            return None                     # catching q would leave no legal way on
        second = weigh(back[1], owner, me, game.rate, self.MOVE)[1]
        for r, (_, count) in zip(second, back[1]):
            rrs += [r] * count
        return out[0] + back[0][1:], tuple(rrs)

    # ---------------------------------------------------- shared arithmetic

    def reserve(self, game, p) -> int:
        """Cash to keep back so the next trip's fees can be paid.

        A few turns of bank fees, plus the opponent fee scaled by how much
        of the map opponents own.
        """
        me = p.index
        theirs, owners = 0, 0
        for o in game.owner:
            if o >= 0 and o != me:
                theirs += 1
                owners |= 1 << o
        if not theirs:
            return self.RESERVE_TURNS * BANK_FEE
        several = 2 if owners & (owners - 1) else 1
        return int(self.RESERVE_TURNS * BANK_FEE + game.rate * several * theirs / len(game.owner))

    def outlook(self, game, p):
        """The parts of a railroad valuation that are the same for every railroad.

        Returns (dollars per unit of share, rivals, cities with at least one
        bank line, cities with at least two).
        """
        leader = max(q.cash for q in game.players if not q.out)
        trips = (game.target_cash - leader) / self.TRIP_INCOME      # trips left in the game
        trips = min(self.HORIZON_MAX, max(self.HORIZON_MIN, trips))
        masks = game.tables.rr_mask
        once = twice = 0
        for r, o in enumerate(game.owner):
            if o < 0:
                twice |= once & masks[r]
                once |= masks[r]
        rivals = [q for q in game.players if q is not p and not q.out]
        return trips * self.FEE_OUTLOOK, rivals, once, twice

    def line_value(self, game, p, r: int, outlook) -> float:
        """What railroad r is worth to player p, in dollars.

        Three parts, each a share of future trips, turned into dollars by
        the expected fee and the number of trips left:

        cover  cities r reaches that p cannot yet reach on their own track.
               Fees p avoids paying.
        lock   for each opponent, cities r reaches that the opponent could
               then enter only on someone's track. Fees others must pay.
        toll   how often shortest routes are forced over r. Fees from
               opponents passing through.
        """
        dollars, rivals, once, twice = outlook
        t = game.tables
        mask = t.rr_mask[r]
        share = self.TOLL_WEIGHT * t.toll[r] * len(rivals)
        uncovered = mask & ~p.access
        if uncovered:
            share += self.COVER_WEIGHT * t.mass(uncovered)
        # Cities left with no bank line once r is in a player's hands.
        closed = mask & ~(twice if game.owner[r] < 0 else once)
        if closed:
            lock = 0.0
            for q in rivals:
                shut = closed & ~q.access
                if shut:
                    lock += t.mass(shut)
            share += self.LOCK_WEIGHT * lock
        return dollars * share


# REACH[d]: chance that two dice total at least d.
REACH = [1.0, 1.0, 1.0] + [sum(1 for a in range(1, 7) for b in range(1, 7) if a + b >= d) / 36 for d in range(3, 13)]


def weigh(runs, owner, me, rate, move):
    """Expected fees for a route, and the railroad to ride on each stretch.

    `runs` is the route as stretches of track with the same choice of
    railroads. The player's own or the bank's track is taken where there is
    a choice; forced onto an opponent, one already being paid is preferred.
    Fees assume every turn covers `move` dots: $1,000 for a turn touching
    own or bank track, and `rate` per opponent per turn.

    Returns (fees, railroad for each run).
    """
    picks = []
    charges = set()             # turn number * 8 + who is paid (0 = bank)
    paying = 0                  # opponents already being paid, as a bit mask
    pos = 0
    for options, count in runs:
        r = options[0]
        o = owner[r]
        if o >= 0 and o != me:
            if len(options) > 1:
                for r2 in options:
                    o2 = owner[r2]
                    if o2 < 0 or o2 == me:
                        r, o = r2, -1
                        break
                else:
                    for r2 in options:
                        if paying >> owner[r2] & 1:
                            r, o = r2, owner[r2]
                            break
            if o >= 0:
                paying |= 1 << o
        else:
            o = -1
        picks.append(r)
        end = pos + count
        first, last = pos // move, (end - 1) // move
        charges.add(first * 8 + o + 1)
        while first < last:
            first += 1
            charges.add(first * 8 + o + 1)
        pos = end
    fees = 0
    for key in charges:
        fees += rate if key & 7 else BANK_FEE
    return fees, picks
