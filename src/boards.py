"""Wordfeud bonus-square boards.

Wordfeud's "random board" mode shuffles the bonus squares each game but keeps
the standard board's 8-fold (D4) symmetry and its per-type counts. We reproduce
that by permuting the bonus-type labels among symmetry orbits of the same size:
positions move, symmetry and counts are preserved.

Codes match the rest of the project: 0 normal, 2 TL, 3 DL, 4 DW, 5 TW.
(1 = a normal tile that currently holds a letter; not used in a bare board.)
"""

from __future__ import annotations

import random

BOARD_SIZE = 15

# Standard Danish Wordfeud layout (centre = double word). D4-symmetric.
# Per-type counts: TW(5)=8, TL(2)=20, DL(3)=20, DW(4)=12 + centre = 13.
STANDARD_BOARD = [
    [3, 0, 0, 0, 5, 0, 0, 2, 0, 0, 5, 0, 0, 0, 3],
    [0, 2, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 2, 0],
    [0, 0, 4, 0, 0, 0, 2, 0, 2, 0, 0, 0, 4, 0, 0],
    [0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0],
    [5, 0, 0, 0, 4, 0, 2, 0, 2, 0, 4, 0, 0, 0, 5],
    [0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0],
    [0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0],
    [2, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 2],
    [0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0],
    [0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 3, 0],
    [5, 0, 0, 0, 4, 0, 2, 0, 2, 0, 4, 0, 0, 0, 5],
    [0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 4, 0, 0, 0, 2, 0, 2, 0, 0, 0, 4, 0, 0],
    [0, 2, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 2, 0],
    [3, 0, 0, 0, 5, 0, 0, 2, 0, 0, 5, 0, 0, 0, 3],
]


def _d4_image(r, c, n=BOARD_SIZE):
    """The 8 dihedral images of cell (r, c)."""
    m = n - 1
    return {
        (r, c), (c, m - r), (m - r, m - c), (m - c, r),    # rotations
        (r, m - c), (m - r, c), (c, r), (m - c, m - r),    # reflections
    }


def d4_orbits(n=BOARD_SIZE):
    """All D4 symmetry orbits of the board, as sorted lists of cells."""
    seen = set()
    orbits = []
    for r in range(n):
        for c in range(n):
            if (r, c) in seen:
                continue
            orb = _d4_image(r, c, n)
            seen |= orb
            orbits.append(sorted(orb))
    return orbits


def random_board(rng=None, n=BOARD_SIZE):
    """A fresh Wordfeud-style board: STANDARD_BOARD's bonus labels permuted among
    orbits of equal size. Symmetry and per-type counts are preserved."""
    rng = rng or random.Random()
    orbits = d4_orbits(n)

    # Each orbit is uniform in STANDARD_BOARD (it is symmetric); group by size.
    by_size = {}
    for orb in orbits:
        code = STANDARD_BOARD[orb[0][0]][orb[0][1]]
        by_size.setdefault(len(orb), []).append((orb, code))

    board = [[0] * n for _ in range(n)]
    for size, group in by_size.items():
        codes = [code for _, code in group]
        rng.shuffle(codes)
        for (orb, _), code in zip(group, codes):
            for (r, c) in orb:
                board[r][c] = code
    return board


def board_counts(board):
    """Count of each code on a board (sanity helper)."""
    out = {}
    for row in board:
        for v in row:
            out[v] = out.get(v, 0) + 1
    return out
