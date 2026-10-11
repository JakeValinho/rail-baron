import json
import math
import statistics

import pytest

from railbaron import Board
from railbaron.layout import even_layout
from railbaron.render import build_payload, render


@pytest.fixture(scope="module")
def board():
    return Board.load()


def test_payload_covers_the_whole_board(board):
    data = build_payload(board)
    assert len(data["railroads"]) == 28 and len(data["cities"]) == 65
    assert sum(len(s["railroads"]) for s in data["segments"]) == 748
    assert len({r["color"] for r in data["railroads"]}) == 28
    assert all({"x", "y", "ex", "ey"} <= set(n) for n in data["nodes"].values())


def test_render_writes_a_page_with_valid_data(board, tmp_path):
    out = render(tmp_path / "map.html", board)
    html = out.read_text(encoding="utf-8")
    assert "__DATA__" not in html
    blob = html.split("const DATA = ", 1)[1].split(";\n", 1)[0]
    assert "Chicago" in {c["name"] for c in json.loads(blob)["cities"]}


def test_even_layout_makes_track_lengths_even(board):
    pos = even_layout(board)
    assert set(pos) == set(board.graph)
    lengths = [math.dist(pos[a], pos[b]) for a, b in board.simple.edges]
    assert statistics.mean(lengths) == pytest.approx(1.0)
    assert 0.6 < min(lengths) and max(lengths) < 1.4
    assert statistics.pstdev(lengths) < 0.1
    # Distance on the page follows distance in moves across the whole map.
    far = math.dist(pos[board.node("Seattle")], pos[board.node("Miami")])
    assert far == pytest.approx(board.distance("Seattle", "Miami"), rel=0.25)


def test_land_background_puts_every_city_on_dry_land(board):
    from railbaron.geography import land_shapes
    from railbaron.layout import geographic_layout

    def inside(p, poly):
        x, y, hit = p[0], p[1], False
        for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
            if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                hit = not hit
        return hit

    shapes = land_shapes(board)
    usa = next(s["points"] for s in shapes if s["kind"] == "land")
    lakes = [s["points"] for s in shapes if s["kind"] == "lake"]
    geo = geographic_layout(board)
    for name, city in board.cities.items():
        assert inside(geo[city.node], usa), name
        assert not any(inside(geo[city.node], lake) for lake in lakes), name
    assert build_payload(board)["land"] == shapes
