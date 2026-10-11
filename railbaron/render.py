"""Draw the board as a self-contained interactive HTML page.

    python -m railbaron.render            writes map.html in the project folder
    python -m railbaron.render --open     and opens it in the browser

The page holds its data as one JSON block, so the game viewer can later be
built by adding game state to the same payload.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import webbrowser
from pathlib import Path

from .board import Board
from .destinations import REGIONS, Destinations
from .geography import land_shapes
from .layout import even_layout, geographic_layout

TEMPLATE = Path(__file__).with_name("map_template.html")
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "map.html"


def railroad_colors(names: list[str]) -> dict[str, str]:
    """A distinct colour per railroad, readable on the dark board."""
    colors = {}
    for i, name in enumerate(names):
        hue = (i * 0.618034) % 1.0  # golden-ratio steps keep neighbours apart
        light = (0.62, 0.74, 0.54)[i % 3]
        sat = (0.85, 0.65, 0.75)[i % 3]
        r, g, b = colorsys.hls_to_rgb(hue, light, sat)
        colors[name] = "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))
    return colors


def build_payload(board: Board, games: list | None = None, open_tab: str | None = None) -> dict:
    colors = railroad_colors(list(board.railroads))
    stats = {row["railroad"]: row for row in board.railroad_stats()}

    # x, y is the geographic map; ex, ey is the even-spacing layout.
    geo, even = geographic_layout(board), even_layout(board)
    nodes = {
        n: {"x": geo[n][0], "y": geo[n][1], "ex": round(even[n][0], 3), "ey": round(even[n][1], 3),
            "region": d["region"]}
        for n, d in board.graph.nodes(data=True)
    }
    segments = [
        {"a": a, "b": b, "railroads": d["railroads"]}
        for a, b, d in board.simple.edges(data=True)
    ]
    railroads = [
        {
            "id": rr,
            "name": info.name,
            "price": info.price,
            "color": colors[rr],
            "cities": stats[rr]["cities"],
            "track": stats[rr]["track"],
            "coverage": stats[rr]["coverage"],
            "traffic_share": stats[rr]["traffic_share"],
        }
        for rr, info in board.railroads.items()
    ]
    cities = [
        {
            "name": c.name,
            "node": c.node,
            "region": c.region,
            "probability": board.destination_probability(c.name),
            "railroads": sorted(board.railroads_at(c.node)),
        }
        for c in board.cities.values()
    ]
    dest = Destinations.load(board=board)
    if games is None:
        from .game import Tables, record_games

        games = record_games(tables=Tables(board, dest))
    destinations = {
        "names": dest.names,
        "nodes": dest.nodes,
        "regions": REGIONS,
        "region": dest.region,
        "region_table": dest.region_table,
        "city_table": dest.city_table,
        "payoff": dest.payoff,
        "richest": dest.richest_region,
    }
    return {"nodes": nodes, "segments": segments, "railroads": railroads, "cities": cities,
            "destinations": destinations, "games": games, "open_tab": open_tab, "land": land_shapes(board)}


def render(out: Path | str = DEFAULT_OUT, board: Board | None = None,
           games: list | None = None, open_tab: str | None = None) -> Path:
    """Write the page. `games` are recordings from railbaron.game; without
    them a fresh game is played for each of 3, 4, 5 and 6 players."""
    board = board or Board.load()
    payload = build_payload(board, games, open_tab)
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", data)
    out = Path(out)
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the Rail Baron map as an HTML page.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="file to write")
    parser.add_argument("--open", action="store_true", help="open the page in the browser")
    args = parser.parse_args()
    out = render(args.out)
    print(f"wrote {out}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
