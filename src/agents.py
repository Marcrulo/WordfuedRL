"""Baseline policies for the side-to-move of a WordfeudGame.

An agent inspects a game and performs one turn (placement, swap, or pass). The
greedy agent is the opponent the RL agent trains against and the baseline to
beat.
"""

from __future__ import annotations

import random


def _fallback(game):
    """No placement available: swap the whole rack if the bag allows, else pass."""
    rack = game.rack()
    if game.bag_count >= game.cfg.rack_size and rack:
        game.swap(list(rack))
    else:
        game.pass_turn()


class GreedyAgent:
    """Plays the highest-scoring legal placement; swaps/passes if none."""

    def act(self, game):
        moves = game.legal_moves()
        if moves:
            game.apply_move(moves[0])      # legal_moves is sorted by score desc
        else:
            _fallback(game)


class RandomAgent:
    """Plays a uniformly random legal placement (for sanity tests / exploration)."""

    def __init__(self, rng=None):
        self.rng = rng or random.Random()

    def act(self, game):
        moves = game.legal_moves()
        if moves:
            game.apply_move(self.rng.choice(moves))
        else:
            _fallback(game)
