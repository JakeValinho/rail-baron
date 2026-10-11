# Rail Baron simulator

Run everything from the project folder. Setup: `pip install -r requirements.txt`

## Commands

### Non-visual

| Command | What it does |
|---|---|
| `python -m railbaron.game --games 5000` | Simulate 5,000 games and print a summary |
| `python -m railbaron.game --games 5000 --workers 0` | Same, on every processor (much faster) |
| `python -m railbaron.game --games 5000 --players 6` | 3 to 6 players (default 4) |
| `python -m railbaron.game --games 5000 --seed 1` | Same dice every run |
| `python -m railbaron.game --choices` | Print the region-choice table |
| `python -m railbaron.destinations` | Destination odds, average payoff, dice speed test |
| `python examples/board_demo.py` | Sample routes, reachable dots, railroad values |
| `python -m pytest` | Run the tests |

### Visual

| Command | What it does |
|---|---|
| `python -m railbaron.game` | Play one recorded game and open it in the browser |
| `python -m railbaron.game --players 6 --seed 7` | Choose player count and dice |
| `python -m railbaron.game --no-open` | Write `map.html` without opening it |
| `python -m railbaron.render --open` | Rebuild and open the map page |
| `start map.html` | Open the existing page without rebuilding |

Links into the page, joined with `&`:

| Link | Opens |
|---|---|
| `map.html#even` | Even-spacing layout |
| `map.html#land` | Land-and-water background |
| `map.html#UP` | A railroad highlighted, by short name |
| `map.html#roll=5` | Destinations tab after five rolls |
| `map.html#game=1&turn=40` | Second recorded game at turn 40 |
| `map.html#play` | Game tab, playing |

## How the code works

Data flows one way: CSV files, then lookup tables, then the engine, then the page.

| File | Job |
|---|---|
| `Map Data/` | Source data: dots, track, cities, destination table, payoff chart, rulebook |
| `railbaron/board.py` | The map as a NetworkX graph, by name. For analysis, not used during play. |
| `railbaron/destinations.py` | Destination dice and payoff chart. Exact odds and numpy batch sampling. |
| `railbaron/tables.py` | Everything numbered as integers for speed, plus the route search. |
| `railbaron/game.py` | **The rules.** Turns, movement, fees, purchases, sales, auctions, winning. |
| `railbaron/policy.py` | **The decisions.** Every choice a player makes, and nothing else. |
| `railbaron/layout.py` | Dot positions: geographic, and even-spacing. |
| `railbaron/geography.py` | Rough outline of the country, bent to fit the board. Background only. |
| `railbaron/render.py` | Packs board, tables and recorded games into `map_template.html`, written out as `map.html`. |
| `tests/` | One test file per module. `test_game.py` replays recorded games and checks every event against the rules. |

Key ideas:

- **One engine, two uses.** The page replays a recording made by the Python engine and applies no rules itself.
- **Rules and decisions are separate.** `game.py` never decides anything. It asks the player's `Policy`.
- **Same seed, same game,** whether recorded or not, and whatever the number of workers.
- **Integers in the hot path.** Dots, railroads and cities are numbers. Sets of railroads are bit masks.
- **`map.html` is generated.** Edit `map_template.html`, then rebuild.

## Player decisions

These are all the ways a player can influence the game. Each is one method in `policy.py`, marked `PLAYER CHOICE`. Everything else is dice.

| # | Decision | Method | Default |
|---|---|---|---|
| 1 | Region, after rolling your own | `choose_region` | Highest average payoff from the current city |
| 2 | Route, and railroad on shared track | `choose_route` | Weighs the shortest route against one detour found by search |
| 3 | Purchase after a payoff | `choose_purchase` | Express at once. Best-value railroad, keeping a cash reserve. Superchief once no railroads are left. |
| 4 | Declare at $200,000, or take another trip first | `choose_declare` | Declares if home is nearer than from a typical city, in turns for the player's train, with less tolerance per opponent near the way. Always declares once an opponent is close to winning. |
| 5 | Forced sale: which line, bank or auction | `choose_sale` | Least valuable line, by auction |
| 6 | Auction bid | `max_bid` | Up to the line's value, never into the reserve |
| 7 | Rover play: chase a declared player | `choose_rover` | Chases when the chance of catching them this turn is worth the detour |

To try another way of playing, subclass `Policy`, override a method or a number, and pass it in:

```python
from railbaron.game import simulate
from railbaron.policy import Policy

class Cautious(Policy):
    RESERVE_TURNS = 8

simulate(2000, players=4, policies=[Cautious(), Policy(), Policy(), Policy()], workers=0)
```

The result's `win_share_by_seat` shows how seat 1 did against the defaults.

## Not in the engine

- The optional rules. A Superchief is always $40,000.
- The roll for who goes first. Seat 1 starts.
