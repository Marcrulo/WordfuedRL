"""Pure Wordfeud game simulator — no vision, no learning.

This is the engine the RL loop runs millions of times: tile bag, two racks,
turn alternation, scoring (via WordfeudEngine), and end-game rules. The vision
pipeline (WordfeudMap) is only for reading a *real* game at deployment; it is
deliberately not imported here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random

from .move_generator import WordfeudEngine, TILE_COUNTS, BOARD_SIZE


@dataclass
class GameConfig:
    rack_size: int = 7
    # Game ends after this many consecutive scoreless turns (passes/swaps).
    # Wordfeud ends a stuck game; the exact count is a tunable here.
    max_scoreless: int = 6


class WordfeudGame:
    """Two-player Wordfeud. Player 0 and player 1 alternate; ``to_move`` says
    whose turn it is. All board/rack letters are lowercase ('*' = blank held in
    a rack; a blank placed on the board is stored as the letter it represents)."""

    def __init__(self, engine: WordfeudEngine, bonus, config: GameConfig = None,
                 tile_counts=None, rng=None):
        self.engine = engine
        self.bonus = [list(row) for row in bonus]
        self.cfg = config or GameConfig()
        self.tile_counts = dict(tile_counts or TILE_COUNTS)
        self.rng = rng or random.Random()
        self.reset()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def reset(self):
        self.bag = []
        for letter, n in self.tile_counts.items():
            self.bag.extend([letter] * n)
        self.rng.shuffle(self.bag)

        self.board = [['' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.racks = [self._draw(self.cfg.rack_size),
                      self._draw(self.cfg.rack_size)]
        self.scores = [0, 0]
        self.to_move = 0
        self.scoreless = 0          # consecutive non-scoring turns
        self.done = False
        self.winner = None          # 0, 1, or None for draw
        self.went_out = None        # player who emptied their rack, if any
        self.board_blanks = set()   # (row, col) of blank tiles played on board
        self.history = []           # list of (player, kind, detail)
        return self

    def _draw(self, n):
        n = min(n, len(self.bag))
        drawn = self.bag[:n]
        self.bag = self.bag[n:]
        return drawn

    # ── queries ────────────────────────────────────────────────────────────

    def legal_moves(self, player=None):
        """Legal placement moves for ``player`` (default: side to move)."""
        p = self.to_move if player is None else player
        return self.engine.legal_moves(self.board, self.bonus, self.racks[p],
                                       self.board_blanks)

    def rack(self, player=None):
        return self.racks[self.to_move if player is None else player]

    @property
    def bag_count(self):
        return len(self.bag)

    # ── actions ──────────────────────────────────────────────────────────────

    def apply_move(self, move):
        """Play a placement Move for the side to move: place tiles, consume rack
        tiles, add score, refill the rack, and pass the turn."""
        if self.done:
            raise RuntimeError("game is over")
        p = self.to_move
        rack = self.racks[p]
        for r, c, ch, is_blank in move.new_tiles:
            if self.board[r][c]:
                raise ValueError(f"square ({r},{c}) already occupied")
            self.board[r][c] = ch
            if is_blank:
                self.board_blanks.add((r, c))
            rack.remove('*' if is_blank else ch)

        self.scores[p] += move.score
        self.history.append((p, 'move', move))
        self.scoreless = 0
        rack.extend(self._draw(self.cfg.rack_size - len(rack)))

        # A player who empties the rack with the bag empty ends the game.
        if not rack and not self.bag:
            self._end(went_out=p)
        else:
            self._next_turn()

    def pass_turn(self):
        if self.done:
            raise RuntimeError("game is over")
        self.history.append((self.to_move, 'pass', None))
        self.scoreless += 1
        self._after_scoreless()

    def swap(self, tiles):
        """Return ``tiles`` (list of rack letters) to the bag and redraw. Allowed
        only when the bag holds at least rack_size tiles (Wordfeud rule)."""
        if self.done:
            raise RuntimeError("game is over")
        if len(self.bag) < self.cfg.rack_size:
            raise ValueError("cannot swap: too few tiles in bag")
        rack = self.racks[self.to_move]
        for t in tiles:
            rack.remove(t)
        rack.extend(self._draw(len(tiles)))
        self.bag.extend(tiles)
        self.rng.shuffle(self.bag)
        self.history.append((self.to_move, 'swap', list(tiles)))
        self.scoreless += 1
        self._after_scoreless()

    # ── turn / end handling ──────────────────────────────────────────────────

    def _next_turn(self):
        self.to_move ^= 1

    def _after_scoreless(self):
        if self.scoreless >= self.cfg.max_scoreless:
            self._end(went_out=None)
        else:
            self._next_turn()

    def _end(self, went_out):
        """Apply end-game rack adjustments and decide the winner."""
        self.done = True
        self.went_out = went_out
        leftovers = [self._rack_points(0), self._rack_points(1)]
        if went_out is not None:
            # Finisher gains the sum of opponents' leftovers; others lose theirs.
            opp = went_out ^ 1
            self.scores[went_out] += leftovers[opp]
            self.scores[opp] -= leftovers[opp]
        else:
            # Stuck game: each player loses their own leftover points.
            for p in (0, 1):
                self.scores[p] -= leftovers[p]
        if self.scores[0] > self.scores[1]:
            self.winner = 0
        elif self.scores[1] > self.scores[0]:
            self.winner = 1
        else:
            self.winner = None

    def _rack_points(self, player):
        return sum(self.engine.points.get(t, 0) for t in self.racks[player])

    def margin(self, player=0):
        """Score margin for ``player`` (positive = winning)."""
        return self.scores[player] - self.scores[player ^ 1]

    # ── debug ──────────────────────────────────────────────────────────────

    def render(self):
        rows = []
        for r in range(BOARD_SIZE):
            rows.append(' '.join((self.board[r][c] or '.') for c in range(BOARD_SIZE)))
        head = (f"P0={self.scores[0]} P1={self.scores[1]} | bag={self.bag_count} | "
                f"to_move={self.to_move} | done={self.done}")
        return head + "\n" + "\n".join(rows)
