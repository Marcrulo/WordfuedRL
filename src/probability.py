"""Hidden-tile probability model.

From one player's view the game is partially observable: you see the board and
your own rack, but not the opponent's rack or the bag order. Everything you
*can't* see is the **unseen pool** = full bag − board tiles − your rack. The
opponent's rack is a random draw from that pool, and so are your future tiles.

These helpers expose that pool and answer "how likely am I (or the opponent) to
hold the tiles to build word W" — by exact expectation for counts and by
Monte-Carlo for whole words (which is robust to blanks acting as wildcards).
"""

from __future__ import annotations

from collections import Counter
import random

from .move_generator import TILE_COUNTS


def unseen_tiles(letters, rack, board_blanks=None, tile_counts=None):
    """The pool of tiles not visible to the player: full bag minus the tiles on
    the board minus the player's own rack. Returned as a Counter (incl. '*').

    ``board_blanks`` (set of (row, col)) marks blank tiles on the board — those
    consume a '*' from the bag, not the letter shown. Where blanks aren't known,
    a board letter exceeding its bag supply is attributed to a blank as a
    fallback."""
    blanks = board_blanks or set()
    pool = Counter(tile_counts or TILE_COUNTS)
    for r, row in enumerate(letters):
        for c, ch in enumerate(row):
            if not ch:
                continue
            if (r, c) in blanks:
                pool['*'] -= 1
            elif pool.get(ch.lower(), 0) > 0:
                pool[ch.lower()] -= 1
            elif pool.get('*', 0) > 0:
                pool['*'] -= 1              # board letter must have been a blank
    for t in rack:
        t = t.lower()
        if pool.get(t, 0) > 0:
            pool[t] -= 1
    return +pool                            # drop zero/negative entries


def can_form(tiles, word):
    """True if ``tiles`` (a Counter incl. '*') can spell ``word``, blanks wild."""
    need = Counter(word)
    shortfall = sum(max(0, n - tiles.get(ch, 0)) for ch, n in need.items())
    return shortfall <= tiles.get('*', 0)


def expected_counts(pool, draw):
    """Expected number of each letter in a random ``draw``-tile sample (linearity
    of expectation: draw * pool[l] / total)."""
    total = sum(pool.values())
    if total == 0:
        return {}
    return {l: draw * n / total for l, n in pool.items()}


def prob_form_word(pool, word, draw=7, trials=4000, rng=None):
    """Monte-Carlo probability that a random ``draw``-tile rack from ``pool`` can
    spell ``word`` (blanks substituting for missing letters)."""
    rng = rng or random.Random()
    flat = list(pool.elements())
    if len(flat) < draw:
        return float(can_form(pool, word))
    word = word.lower()
    hits = 0
    for _ in range(trials):
        hand = Counter(rng.sample(flat, draw))
        hits += can_form(hand, word)
    return hits / trials


def draw_rack(pool, k, rng=None):
    """Sample a ``k``-tile rack (list of letters) from ``pool`` without replacement."""
    rng = rng or random.Random()
    flat = list(pool.elements())
    if len(flat) <= k:
        return flat
    return rng.sample(flat, k)
