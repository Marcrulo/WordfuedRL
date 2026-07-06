"""Wordfeud move generation, scoring, and state transitions.

Given a board (placed letters + bonus squares) and a rack of letters, this
finds every legal move, scores it, and produces the resulting board state —
i.e. the (action, next_state) pairs an RL agent needs.

Algorithm is the classic Scrabble generator: anchor squares + a trie of the
dictionary + cross-checks, run once horizontally and once on the transposed
board for vertical moves.

Coordinate convention: grids are indexed [row][col], row 0 = top, col 0 = left.
Letters are stored lowercase ('' = empty). Blanks in the rack are '*' and
score 0 points.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter

BOARD_SIZE = 15
BINGO_BONUS = 40          # Wordfeud bonus for using all 7 rack tiles (Scrabble = 50)
CENTER = (BOARD_SIZE // 2, BOARD_SIZE // 2)

# Default Danish Wordfeud letter values (lowercase keys). Blank '*' = 0.
DEFAULT_POINTS = {
    'a': 1, 'b': 3, 'c': 8, 'd': 2, 'e': 1, 'f': 3, 'g': 3, 'h': 4,
    'i': 3, 'j': 4, 'k': 3, 'l': 2, 'm': 4, 'n': 1, 'o': 2, 'p': 4,
    'r': 1, 's': 2, 't': 2, 'u': 3, 'v': 4, 'x': 8, 'y': 4, 'z': 9,
    'æ': 4, 'ø': 4, 'å': 4, '*': 0,
}

# Danish Wordfeud tile bag: letter -> count. 103 letters + 2 blanks = 105 tiles.
TILE_COUNTS = {
    'a': 7, 'b': 4, 'c': 2, 'd': 5, 'e': 9, 'f': 3, 'g': 3, 'h': 2,
    'i': 4, 'j': 2, 'k': 4, 'l': 5, 'm': 3, 'n': 7, 'o': 5, 'p': 2,
    'r': 7, 's': 6, 't': 6, 'u': 3, 'v': 4, 'x': 1, 'y': 2, 'z': 1,
    'æ': 2, 'ø': 2, 'å': 2, '*': 2,
}

# Bonus-square code -> (letter multiplier, word multiplier).
# Verified against the in-app labels (green=DL, blue=TL, orange=DW, red=TW):
#   0 dark / 1 normal -> no bonus
#   2 green  = double letter (DL, letter x2)
#   3 blue   = triple letter (TL, letter x3)
#   4 orange = double word   (DW, word x2)
#   5 red    = triple word   (TW, word x3)
BONUS_MULT = {0: (1, 1), 1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (1, 2), 5: (1, 3)}

# Bonus-square colours (RGB 0-1), matching the board's own palette.
BONUS_COLOR = {
    0: (0.16, 0.17, 0.19), 1: (0.92, 0.91, 0.87), 2: (0.46, 0.55, 0.37),
    3: (0.21, 0.38, 0.54), 4: (0.72, 0.47, 0.20), 5: (0.51, 0.20, 0.19),
}
BONUS_LABEL = {2: 'DL', 3: 'TL', 4: 'DW', 5: 'TW'}


def plot_board(letters, bonus, move=None, ax=None, title=None, board_blanks=None):
    """Draw the board: bonus squares coloured, placed letters shown. If a
    ``move`` is given, its new tiles are highlighted (gold) and the rest of the
    letters dimmed. ``board_blanks`` (set of (row, col)) renders blank tiles in
    grey (they score 0). Returns the matplotlib Axes."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    letters = [[(c or '').lower() for c in row] for row in letters]
    n = len(letters)
    blanks = set(board_blanks or ())
    new = {(r, c): (ch, b) for r, c, ch, b in (move.new_tiles if move else [])}

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    for r in range(n):
        for c in range(n):
            code = int(bonus[r][c]) if str(bonus[r][c]).isdigit() else 1
            ax.add_patch(Rectangle((c, n - 1 - r), 1, 1,
                                   facecolor=BONUS_COLOR.get(code, BONUS_COLOR[1]),
                                   edgecolor='white', linewidth=1))
            if code in BONUS_LABEL and not letters[r][c] and (r, c) not in new:
                ax.text(c + 0.5, n - 1 - r + 0.5, BONUS_LABEL[code], ha='center',
                        va='center', fontsize=7, color='white', alpha=0.8)

    for r in range(n):
        for c in range(n):
            if (r, c) in new:
                ch, blank = new[(r, c)]
                ax.add_patch(Rectangle((c, n - 1 - r), 1, 1, facecolor='gold',
                                       edgecolor='black', linewidth=2, zorder=2))
                ax.text(c + 0.5, n - 1 - r + 0.5, ch.upper(), ha='center',
                        va='center', fontsize=14, fontweight='bold',
                        color=('gray' if blank else 'black'), zorder=3)
            elif letters[r][c]:
                is_blank = (r, c) in blanks
                ax.add_patch(Rectangle((c, n - 1 - r), 1, 1, facecolor='#efe8c8',
                                       edgecolor='white', linewidth=1, zorder=1))
                ax.text(c + 0.5, n - 1 - r + 0.5, letters[r][c].upper(),
                        ha='center', va='center', fontsize=13,
                        fontstyle=('italic' if is_blank else 'normal'),
                        color=('lightgray' if is_blank else
                               'dimgray' if move else 'black'), zorder=2)

    ax.set_xlim(0, n); ax.set_ylim(0, n)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title)
    return ax


def split_board(letter_map):
    """Split a ``WordfeudMap.map_with_letters()`` grid into (letters, bonus).

    ``map_with_letters`` stores recognised letters on normal tiles and the
    bonus-square code (as a string) elsewhere. This returns a clean letters
    grid ('' = empty) and an int bonus grid for the engine.

    NOTE: letter recognition currently has no confidence threshold, so empty
    normal tiles may carry a spurious letter — filter those upstream first.
    """
    n = len(letter_map)
    letters = [['' for _ in range(n)] for _ in range(n)]
    bonus = [[1 for _ in range(n)] for _ in range(n)]
    for r in range(n):
        for c in range(n):
            cell = letter_map[r][c]
            s = '' if cell is None else str(cell)
            if s.isalpha():
                letters[r][c] = s.lower()
            elif s.isdigit():
                bonus[r][c] = int(s)
    return letters, bonus


@dataclass
class Move:
    """One legal play. ``new_tiles`` are the (row, col, letter, is_blank) tiles
    placed this turn; ``score`` includes all formed words and the bingo bonus."""
    new_tiles: list                      # [(r, c, letter, is_blank), ...]
    score: int
    word: str = ''                       # main (longest) word formed, for display
    direction: str = ''                  # 'H' or 'V'
    start: tuple = (0, 0)                # (row, col) of main word start

    def rack_used(self):
        return [('*' if b else ch) for _, _, ch, b in self.new_tiles]

    def __repr__(self):
        return (f"Move({self.word!r} {self.direction} @{self.start} "
                f"score={self.score} tiles={len(self.new_tiles)})")


class WordfeudEngine:
    """Holds the dictionary; generates/scores moves for any board + rack."""

    def __init__(self, words, points=None, bingo_bonus=BINGO_BONUS):
        self.points = dict(points) if points else dict(DEFAULT_POINTS)
        self.bingo_bonus = bingo_bonus
        self.alphabet = [c for c in self.points if c != '*']
        self.word_set = set()
        self.trie = {}
        for w in words:
            w = w.lower()
            self.word_set.add(w)
            node = self.trie
            for ch in w:
                node = node.setdefault(ch, {})
            node['$'] = True

    # ── public API ────────────────────────────────────────────────────────

    def legal_moves(self, letters, bonus, rack, board_blanks=None):
        """All legal moves. ``letters``/``bonus`` are 15x15 grids; ``rack`` a
        list of uppercase/lowercase letters ('*' for blank). ``board_blanks`` is
        an optional set of (row, col) for blank tiles already on the board (they
        score 0 in words built across them). Returns scored, deduped Moves
        sorted by descending score."""
        letters = [[(c or '').lower() for c in row] for row in letters]

        # Generate raw new-tile lists (in original board coords) both axes.
        raw = self._gen_axis(letters, bonus, rack, transposed=False)
        tL, tB = self._transpose(letters), self._transpose(bonus)
        raw += self._gen_axis(tL, tB, rack, transposed=True)

        # Dedupe identical plays, then score each on the original board.
        moves = {}
        for new_tiles in raw:
            key = frozenset((r, c, ch) for r, c, ch, _ in new_tiles)
            if key in moves:
                continue
            score, word, direction, start = self._score(
                letters, bonus, new_tiles, board_blanks)
            moves[key] = Move(new_tiles, score, word, direction, start)
        return sorted(moves.values(), key=lambda mv: -mv.score)

    def next_state(self, letters, move):
        """Return a new letters grid with ``move`` applied (the next state)."""
        new = [[(c or '').lower() for c in row] for row in letters]
        for r, c, ch, _ in move.new_tiles:
            new[r][c] = ch
        return new

    def score_move(self, letters, bonus, new_tiles, board_blanks=None):
        """Score a set of new tiles on a board. Returns (score, word, dir, start)."""
        return self._score(
            [[(c or '').lower() for c in row] for row in letters], bonus,
            new_tiles, board_blanks)

    # ── generation (single orientation) ─────────────────────────────────────

    def _gen_axis(self, letters, bonus, rack, transposed):
        rack_counts = Counter(c.lower() for c in rack)
        cross = self._cross_checks(letters)
        anchors = self._anchors(letters)
        results = []

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if (r, c) not in anchors:
                    continue
                node = self.trie
                if c > 0 and letters[r][c - 1]:
                    # Left part is fixed: walk the existing prefix in the trie.
                    sc = c
                    while sc > 0 and letters[r][sc - 1]:
                        sc -= 1
                    prefix = ''.join(letters[r][sc:c])
                    node = self._walk(prefix)
                    if node is not None:
                        self._extend_right(letters, bonus, cross, rack_counts,
                                           r, c, c, node, [], [], results, transposed)
                else:
                    # Build optional left part from the rack, bounded by the
                    # empty run to the left (stop before another anchor).
                    limit = 0
                    cc = c - 1
                    while cc >= 0 and not letters[r][cc] and (r, cc) not in anchors:
                        limit += 1
                        cc -= 1
                    self._left_part(letters, bonus, cross, rack_counts,
                                    r, c, self.trie, [], limit, results, transposed)
        return results

    def _left_part(self, L, B, cross, rack, r, anchor_c, node, left, limit,
                   out, transposed):
        # left = [(letter, is_blank), ...] new tiles placed left of the anchor.
        self._extend_right(L, B, cross, rack, r, anchor_c, anchor_c, node,
                           left, [], out, transposed)
        if limit <= 0:
            return
        for ch in list(node):
            if ch == '$':
                continue
            for letter, is_blank in self._rack_options(rack, ch):
                rack[letter] -= 1
                self._left_part(L, B, cross, rack, r, anchor_c, node[ch],
                                left + [(ch, is_blank)], limit - 1, out, transposed)
                rack[letter] += 1

    def _extend_right(self, L, B, cross, rack, r, c, anchor_c, node, left,
                      right, out, transposed):
        # right = [(col, letter, is_blank, is_new), ...] from anchor rightward.
        new_count = len(left) + sum(1 for _, _, _, isnew in right if isnew)
        # Record only once the span has covered the anchor (c past anchor_c),
        # otherwise the word wouldn't connect to the anchor square.
        if (c >= BOARD_SIZE or not L[r][c]) and '$' in node and new_count > 0 \
                and c > anchor_c:
            self._record(B, r, anchor_c, left, right, out, transposed)
        if c >= BOARD_SIZE:
            return

        cell = L[r][c]
        if cell:                                    # square already has a tile
            if cell in node:
                self._extend_right(L, B, cross, rack, r, c + 1, anchor_c,
                                   node[cell], left,
                                   right + [(c, cell, False, False)],
                                   out, transposed)
        else:                                       # empty: try rack tiles
            allowed = cross[r][c]                   # None = any letter ok
            for ch in list(node):
                if ch == '$':
                    continue
                if allowed is not None and ch not in allowed:
                    continue
                for letter, is_blank in self._rack_options(rack, ch):
                    rack[letter] -= 1
                    self._extend_right(L, B, cross, rack, r, c + 1, anchor_c,
                                       node[ch], left,
                                       right + [(c, ch, is_blank, True)],
                                       out, transposed)
                    rack[letter] += 1

    def _record(self, B, r, anchor_c, left, right, out, transposed):
        # Left tiles occupy columns anchor_c-len(left) .. anchor_c-1.
        start = anchor_c - len(left)
        new_tiles = []
        for i, (ch, blank) in enumerate(left):
            new_tiles.append((r, start + i, ch, blank))
        for col, ch, blank, isnew in right:
            if isnew:
                new_tiles.append((r, col, ch, blank))
        if transposed:                              # map (row, col) back: swap
            new_tiles = [(c, rr, ch, blank) for rr, c, ch, blank in new_tiles]
        out.append(new_tiles)

    # ── scoring ─────────────────────────────────────────────────────────────

    def _score(self, L, B, new_tiles, board_blanks=None):
        board_blanks = board_blanks or set()
        temp = [row[:] for row in L]
        new_pos = {}
        for r, c, ch, blank in new_tiles:
            temp[r][c] = ch
            new_pos[(r, c)] = blank

        total = 0
        seen = set()
        main = None
        for (r, c, ch, _) in new_tiles:
            for dr, dc in ((0, 1), (1, 0)):
                sr, scc = r, c
                while self._inb(sr - dr, scc - dc) and temp[sr - dr][scc - dc]:
                    sr -= dr; scc -= dc
                cells = []
                rr, cc = sr, scc
                while self._inb(rr, cc) and temp[rr][cc]:
                    cells.append((rr, cc)); rr += dr; cc += dc
                if len(cells) < 2:                       # single tile = no word
                    continue
                if not any((cr, cc2) in new_pos for cr, cc2 in cells):
                    continue                             # word has no new tile
                key = (cells[0], dr, dc)
                if key in seen:
                    continue
                seen.add(key)
                wscore, wmult = 0, 1
                for (cr, cc2) in cells:
                    base = self.points.get(temp[cr][cc2], 0)
                    if (cr, cc2) in new_pos:
                        if new_pos[(cr, cc2)]:
                            base = 0
                        lm, wm = BONUS_MULT.get(B[cr][cc2], (1, 1))
                        wscore += base * lm
                        wmult *= wm
                    else:
                        wscore += 0 if (cr, cc2) in board_blanks else base
                total += wscore * wmult
                word = ''.join(temp[cr][cc2] for cr, cc2 in cells)
                if main is None or len(word) > len(main[0]):
                    direction = 'H' if dc else 'V'
                    main = (word, direction, cells[0])

        if len(new_tiles) == 7:
            total += self.bingo_bonus
        if main is None:
            return total, '', '', (0, 0)
        return total, main[0], main[1], main[2]

    # ── helpers ──────────────────────────────────────────────────────────────

    def _cross_checks(self, L):
        """For each empty cell, the set of letters that form a legal vertical
        (cross-direction) word, or None if the cell has no vertical neighbour."""
        cross = [[None] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if L[r][c]:
                    continue
                up = ''
                rr = r - 1
                while rr >= 0 and L[rr][c]:
                    up = L[rr][c] + up; rr -= 1
                down = ''
                rr = r + 1
                while rr < BOARD_SIZE and L[rr][c]:
                    down += L[rr][c]; rr += 1
                if not up and not down:
                    cross[r][c] = None
                else:
                    cross[r][c] = {ch for ch in self.alphabet
                                   if (up + ch + down) in self.word_set}
        return cross

    def _anchors(self, L):
        """Empty squares adjacent to a placed tile; the centre if board empty."""
        any_tile = any(L[r][c] for r in range(BOARD_SIZE) for c in range(BOARD_SIZE))
        if not any_tile:
            return {CENTER}
        anchors = set()
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if L[r][c]:
                    continue
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    if self._inb(r + dr, c + dc) and L[r + dr][c + dc]:
                        anchors.add((r, c)); break
        return anchors

    def _rack_options(self, rack, ch):
        """Yield (rack_key, is_blank) ways to play letter ``ch`` from the rack."""
        if rack.get(ch, 0) > 0:
            yield ch, False
        if rack.get('*', 0) > 0:
            yield '*', True

    def _walk(self, prefix):
        node = self.trie
        for ch in prefix:
            node = node.get(ch)
            if node is None:
                return None
        return node

    @staticmethod
    def _inb(r, c):
        return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

    @staticmethod
    def _transpose(grid):
        return [list(row) for row in zip(*grid)]
