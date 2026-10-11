"""Layouts for drawing the board.

geographic_layout() puts every dot where the printed map has it.

even_layout() ignores geography. It places dots so that the straight-line
distance between any two of them is as close as possible to the number of
moves between them, which makes every piece of track about the same length.
Reading distance off that picture is reading the real cost of a trip.
"""

from __future__ import annotations

import numpy as np

from .board import Board

# The source map was drawn for a screen with tall pixels. Stretching y by
# this much gives the country its familiar shape.
Y_STRETCH = 1.8


def geographic_layout(board: Board, y_stretch: float = Y_STRETCH) -> dict[str, tuple[float, float]]:
    return {n: (float(d["x"]), d["y"] * y_stretch) for n, d in board.graph.nodes(data=True)}


def even_layout(board: Board, iterations: int = 120) -> dict[str, tuple[float, float]]:
    """Positions where distance on the page tracks distance in moves.

    One piece of track comes out about 1 unit long. The layout starts from
    the geographic map, so west stays on the left and north at the top.

    Method: stress majorisation. Each pair of dots wants to sit as far apart
    as the number of moves between them; nearby pairs count for more, and
    dots joined by track count for most, so track lengths come out even.
    The result is kept on the board, so asking twice costs nothing.
    """
    cached = getattr(board, "_even_layout", None)
    if cached is not None and cached[0] == iterations:
        return cached[1]
    nodes = list(board.graph)
    n = len(nodes)
    moves = board._distances
    dist = np.array([[moves[a][b] for b in nodes] for a in nodes], dtype=float)
    off = ~np.eye(n, dtype=bool)

    weight = np.zeros((n, n))
    weight[off] = dist[off] ** -3.0
    weight[dist == 1] *= 5.0
    # Solving for new positions needs the inverse of this matrix, once.
    lap = np.diag(weight.sum(axis=1)) - weight
    lap_inv = np.linalg.pinv(lap)

    geo = geographic_layout(board)
    pos = np.array([geo[node] for node in nodes])
    edges = np.array([(nodes.index(a), nodes.index(b)) for a, b in board.simple.edges])
    pos = pos / np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1).mean()

    pull = weight * dist
    for _ in range(iterations):
        gap = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        gap[~off] = 1.0
        b = -pull / gap
        b[~off] = 0.0
        b[np.diag_indices(n)] = -b.sum(axis=1)
        pos = lap_inv @ (b @ pos)

    pos /= np.linalg.norm(pos[edges[:, 0]] - pos[edges[:, 1]], axis=1).mean()
    pos -= pos.min(axis=0)
    layout = {node: (float(x), float(y)) for node, (x, y) in zip(nodes, pos)}
    board._even_layout = (iterations, layout)
    return layout
