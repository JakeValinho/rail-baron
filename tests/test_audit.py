"""Rulebook-derived regressions, independent of the whole-game replay checker."""
import json
import csv
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

from railbaron.board import Board, Railroad
from railbaron.game import Game, TARGET_CASH
from railbaron.policy import Policy
from railbaron.tables import Tables, search


@pytest.fixture(scope='module')
def tables():
    return Tables()


class NoChoices(Policy):
    def choose_purchase(self, game, p):
        return None

    def choose_rover(self, game, p, q, blocked):
        return None


def trip(g, p, start='New York', end='Chicago'):
    p.city = p.trip_from = g.dest.index(start)
    p.destination = g.dest.index(end)
    p.dots, _, _, p.rrs = g.tables.shortest(g.tables.city_node[p.city], g.tables.city_node[p.destination])
    p.step, p.goal = 0, p.dots[-1]


def test_declared_mover_is_not_caught_but_can_catch_a_stationary_declared_pawn(tables):
    g = Game(3, 1, tables, policies=NoChoices())
    p, q = g.players[:2]
    for player in (p, q):
        trip(g, player)
        player.declared = True
        player.cash = TARGET_CASH + 60000
    q.step = 2
    g.declared = 2
    g._move(p, 3, [], [1, 2])
    assert not q.declared and q.cash == TARGET_CASH + 10000
    assert p.declared and p.cash == TARGET_CASH + 110000


def test_declaring_at_home_on_bonus_waits_for_outstanding_fees(tables):
    g = Game(3, 1, tables, policies=NoChoices())
    p = g.players[0]
    trip(g, p, 'Philadelphia', 'New York')
    p.home = p.destination
    p.cash = TARGET_CASH - g.dest.payoff[p.city][p.destination]
    g.rand = lambda: 35 / 36  # boxcars: arrive, then start a bonus trip
    g.play_turn()
    assert p.cash == TARGET_CASH - 1000
    assert g.winner is None


def test_compressed_search_checks_internal_used_track():
    # A compressed a--b--c--d stretch must not jump over used b--c.
    adj = (((1, (0,), 1, (), 1),),
           ((0, (0,), 1, (), 1), (2, (0,), 1, (), 1)),
           ((1, (0,), 1, (), 1), (3, (0,), 1, (), 1)),
           ((2, (0,), 1, (), 1),))
    t = SimpleNamespace(adj=adj, kadj=[((3, (0,), 3, (1, 2), 1),), None, None,
                                     ((0, (0,), 3, (2, 1), 1),)],
                        hops=[[abs(i-j) for j in range(4)] for i in range(4)])
    assert search(t, 0, 3, 1, 1, 0, {(1, 2, 0), (2, 1, 0)}) is None


def small_board(edges):
    graph = nx.MultiGraph()
    for a, b, r in edges:
        graph.add_edge(a, b, key=r, railroad=r)
    return Board(graph, {r: Railroad(r, r, 4000) for _, _, r in edges}, {})


def test_reachable_can_return_on_the_second_parallel_railroad():
    b = small_board([('a', 'b', 'R'), ('a', 'b', 'S')])
    assert set(b.reachable('a', 2)) == {'a'}


def test_analysis_moves_cannot_end_trapped_away_from_destination():
    b = small_board([('a', 'b', 'R'), ('a', 'c', 'R')])
    assert [r.end for r in b.legal_moves('a', 1, 'c')] == ['c']
    assert set(b.reachable('a', 1, 'c')) == {'c'}


def test_policy_does_not_reuse_track_on_an_ordinary_trip(tables):
    g = Game(3, 1, tables)
    p = g.players[0]
    trip(g, p)
    src = p.dot
    blocked = {(src, edge[0], r) for edge in tables.adj[src] for r in edge[1]}
    blocked |= {(b, a, r) for a, b, r in list(blocked)}
    with pytest.raises(ValueError, match='legal route'):
        p.policy.choose_route(g, p, src, p.goal, blocked)


def js_functions(*names):
    text = Path('railbaron/map_template.html').read_text(encoding='utf-8')
    pieces = []
    for name in names:
        begin = text.index('function ' + name + '(')
        # All these functions end at a line-start closing brace.
        end = text.index('\n}', begin) + 2
        pieces.append(text[begin:end])
    return '\n'.join(pieces)


def run_js(source):
    if shutil.which('node') is None:
        pytest.skip('Node is needed to verify the browser replay')
    result = subprocess.run(['node', '-e', source], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_browser_region_average_includes_lost_turns():
    run_js("const assert = require('node:assert/strict');\n" +
           "const D = {city_table:[[0,1]],nodes:['a','b'],payoff:[[0,10000]]};\n" +
           js_functions('regionValue') + '\nassert.equal(regionValue(0,0),5000);')


def test_browser_replay_keeps_declaration_at_home_alternate():
    run_js("const assert = require('node:assert/strict');\n" + js_functions('applyEvent') +
           "\nconst st={players:[{home:1,dest:1,declared:true,cash:200000,trips:0}],owner:[]};" +
           "applyEvent(st,0,{t:'arrive',city:1,pay:1000});assert.equal(st.players[0].declared,true);")


def test_recorded_bankruptcy_reports_actual_payments(tables):
    g = Game(3, 1, tables, policies=NoChoices(), record=True)
    p = g.players[0]
    p.cash = 7000
    g.due, g.bank_due = {1: 5000, 2: 5000}, 1000
    events = []
    g._settle(p, events)
    fees = next(e for e in events if e['t'] == 'fees')
    assert fees['to'] == [(1, 5000), (2, 2000)]
    assert fees['bank'] == 0


def test_caught_pawn_purchase_changes_fees_for_subsequent_steps(tables):
    g = Game(3, 1, tables, policies=NoChoices())
    p, q = g.players[:2]
    trip(g, p)
    r = tables.rr_ids.index('NYC')

    class BuyNYC(NoChoices):
        def choose_purchase(self, game, player):
            return r

    q.policy = BuyNYC()
    q.city = q.trip_from = g.dest.index('Boston')
    q.destination = g.dest.index('Albany')
    q.dots, q.rrs, q.step = (tables.city_node[q.destination],), (), 0
    q.goal = tables.city_node[q.home]
    q.cash, q.declared, g.declared = 260000, True, 1
    g._move(p, 4, [], [2, 2])
    assert g.owner[r] == q.index
    assert g.bank_due == 1000
    assert g.due == {q.index: 1000}  # established at Albany when q bought NYC


def test_repurchase_from_bank_uses_the_current_old_bank_fee(tables):
    g = Game(3, 1, tables, policies=NoChoices())
    p, q, buyer = g.players
    trip(g, p)
    r = p.rrs[0]
    g._transfer(r, q.index)
    p.est = {r: 5000}  # an established rate from the all-sold increase
    g._transfer(r, -1)
    g._transfer(r, buyer.index)
    p.replan = False
    g._move(p, 1, [], [1])
    assert g.due == {buyer.index: 1000}


def test_batch_destination_rolls_do_not_change_city_on_a_lost_twin_roll(tables):
    d = tables.destinations
    start = d.index('Minneapolis')
    cities, payoffs = d.sample_trips(50000, 2, start=start, seed=7)
    for first, second, pay in zip(cities[:, 0], cities[:, 1], payoffs[:, 1]):
        if d.nodes[first] == d.nodes[start]:
            assert pay == d.payoff[start][second]


def test_long_run_payoff_per_trip_excludes_lost_destination_rolls():
    from railbaron.destinations import Destinations
    d = Destinations(['a', 'b'], ['a', 'b'], [0, 0], [0]*72,
                     [[0]*54 + [1]*18]*7, [[0, 10000], [10000, 0]])
    _, per_trip = d.long_run()
    assert per_trip == pytest.approx(10000)


def test_transition_matrix_keeps_the_original_twin_city(tables):
    d = tables.destinations
    a, b = d.index('Minneapolis'), d.index('St Paul')
    T = d.transition_matrix()
    odds = d.destination_odds(a)
    assert T[a, b] == 0
    assert T[a, a] == pytest.approx(odds[a] + odds[b])


def test_alternate_route_reuses_the_minimum_homeward_track(tables):
    g = Game(3, 1, tables, policies=NoChoices())
    p = g.players[0]
    edges = [(0, 1), (1, 2), (0, 3), (3, 4), (4, 2)]
    graph = nx.Graph(edges)
    adj = [tuple((b, (0,), 1, (), 1) for b in graph[a]) for a in range(5)]
    g.tables = SimpleNamespace(adj=adj)
    p.reusable = {(0, 1, 0), (1, 2, 0), (2, 4, 0)}
    blocked = {(a, b, 0) for a, b, _ in p.reusable}
    blocked |= {(b, a, r) for a, b, r in list(blocked)}
    dots, rrs = g._reuse_route(p, 0, 2, blocked)
    assert dots == (0, 3, 4, 2)  # three dots but one reuse, rather than two reused sections
    g._validate_route(p, dots, rrs, 0, 2, blocked)
    with pytest.raises(ValueError, match='more track than necessary'):
        g._validate_route(p, (0, 1, 2), (0, 0), 0, 2, blocked)
    p.reusable.clear()
    with pytest.raises(ValueError, match='legal route'):
        g._reuse_route(p, 0, 2, blocked)


def test_engine_rejects_custom_routes_that_pass_the_destination(tables):
    g = Game(3, 1, tables, policies=NoChoices())
    p = g.players[0]
    trip(g, p)
    dots, rrs = p.dots, p.rrs
    with pytest.raises(ValueError, match='legal route'):
        g._validate_route(p, dots + (dots[-2], dots[-1]), rrs + (rrs[-1], rrs[-1]), dots[0], dots[-1], None)


def test_all_216_destination_dice_match_the_python_and_js_tables(tables):
    from railbaron.destinations import outcome
    d = tables.destinations
    for k in range(216):
        black, white1, white2 = k // 36 + 1, k // 6 % 6 + 1, k % 6 + 1
        o = outcome(black, white1, white2)
        assert tables.region_by_roll[k] == d.region_table[o]
        assert all(tables.city_by_roll[r][k] == d.city_table[r][o] for r in range(7))
    source = Path('railbaron/map_template.html').read_text(encoding='utf-8')
    js = source.split('const outcome = ', 1)[1].split(';', 1)[0]
    run_js("const assert=require('node:assert/strict');const outcome=" + js + ';' +
           'const expected=' + json.dumps([outcome(k//36+1,k//6%6+1,k%6+1) for k in range(216)]) + ';' +
           'for(let k=0;k<216;k++)assert.equal(outcome(Math.floor(k/36)+1,Math.floor(k/6)%6+1,k%6+1),expected[k]);')


def test_browser_replays_every_cash_event_without_corrections(tables):
    games = [Game(n, 100*n+seed, tables, record=True).play().recording
             for n in (3, 4, 5, 6) for seed in range(2)]
    code = "const assert=require('node:assert/strict');" + js_functions('applyEvent') + '''
    let input='';process.stdin.on('data',s=>input+=s);process.stdin.on('end',()=>{
      for(const g of JSON.parse(input)) {
        const st={owner:Array(28).fill(-1),players:g.players.map(()=>({cash:g.start_cash,trips:0})),winner:null};
        for(const turn of g.turns) {
          for(const e of turn.ev)applyEvent(st,turn.p,e);
          assert.deepEqual(st.players.map(p=>p.cash),turn.cash);
        }
        assert.equal(st.winner,g.winner);
      }
    });'''
    if shutil.which('node') is None:
        pytest.skip('Node is needed to verify the browser replay')
    result = subprocess.run(['node', '-e', code], input=json.dumps(games), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_destination_csv_cell_by_cell_against_rulebook():
    rules = Path('Map Data/railbaron_rules_cleaned.txt').read_text(encoding='utf-8')
    text = rules.split('DESTINATION TABLE (BACK OF RULEBOOK)', 1)[1].split('RAILROAD PRICE CHART', 1)[0]
    labels = ['REGION', 'NORTHEAST', 'SOUTHEAST', 'NORTH CENTRAL', 'SOUTH CENTRAL', 'PLAINS', 'NORTHWEST', 'SOUTHWEST']
    parts = re.split(r'\n(' + '|'.join(labels) + r')\n', text)
    expected = {}
    for section, body in zip(parts[1::2], parts[2::2]):
        for line in body.splitlines():
            match = re.match(r'^(\d+)\s+(.+?)\s{2,}(.+?)\s*$', line)
            if match:
                total, odd, even = match.groups()
                for parity, result in [('odd', odd), ('even', even)]:
                    expected[section, int(total), parity] = result
    names = dict(zip(['', 'NorthEast', 'SouthEast', 'NorthCentral', 'SouthCentral', 'Plains', 'NorthWest', 'SouthWest'], labels))
    def norm(value):
        return (value.upper().replace('.', '').replace(' ', '')
                .replace('OKLCITY', 'OKLAHOMACITY').replace('WASHINGTONDC', 'WASHINGTON'))
    rows = list(csv.DictReader(Path('Map Data/rail_baron_destinations.csv').open(encoding='utf-8-sig')))
    assert len(rows) == len(expected) == 176
    seen = set()
    for row in rows:
        key = (names[row['region']], int(row['dice_sum']), row['black_die_parity'])
        assert key not in seen
        seen.add(key)
        actual = row['result']
        if actual in ('Portland ME', 'Portland OR'):
            actual = 'Portland'
        assert norm(actual) == norm(expected[key]), key


def test_railroad_prices_against_rulebook(tables):
    text = Path('Map Data/railbaron_rules_cleaned.txt').read_text(encoding='utf-8').split('RAILROAD PRICE CHART (BACK OF RULEBOOK)', 1)[1]
    expected = {code: int(price.replace(',', ''))
                for code, price in re.findall(r'^([A-Z&]+)\s+.+?\$([\d,]+)', text, re.M)}
    assert len(expected) == len(tables.rr_ids) == 28
    for code, price in zip(tables.rr_ids, tables.rr_price):
        assert price == expected[code.upper().replace('+', '&')]


def test_payoff_shape_labels_symmetry_diagonal_and_same_dot_zero(tables):
    rows = list(csv.reader(Path('Map Data/Payout.csv').open(encoding='utf-8-sig')))
    assert len(rows) == 68 and all(len(row) == 68 for row in rows)
    assert rows[0][1:] == [row[0] for row in rows[1:]]
    d = tables.destinations
    for a in range(d.count):
        for b in range(d.count):
            assert d.payoff[a][b] == d.payoff[b][a]
            if d.nodes[a] == d.nodes[b]:
                assert d.payoff[a][b] == 0


def test_purchase_cannot_use_bank_sentinel_as_a_railroad_index(tables):
    class InvalidPurchase(NoChoices):
        def choose_purchase(self, game, p):
            return -1
    g = Game(3, 1, tables, policies=InvalidPurchase())
    p = g.players[0]
    trip(g, p, 'Chicago', 'Miami')
    p.cash = 100000
    with pytest.raises(ValueError, match='railroad'):
        g._arrive(p, [])
