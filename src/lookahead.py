"""2-ply Monte-Carlo lookahead agent (Quackle/Maven-style simulation).

Instead of learning a value function, this evaluates each of its candidate moves
by simulating the opponent's reply over racks sampled from the unseen-tile pool:

    value(m) = m.score - E_{opp rack ~ unseen}[ best opponent reply on board+m ]
                       + leave_weight * leave_value(my leftover rack)

The expectation captures *defense* for free — a move that opens a triple-word
square scores well now but inflates the opponent's expected reply, lowering its
value. No training required; strength scales with the number of samples.

Cost: for K candidate moves and N opponent samples it runs K·N legal-move
generations per turn, so keep K and N modest (this is an inference-time search).
"""

from __future__ import annotations

import random
from collections import Counter

from .agents import _fallback
from .probability import unseen_tiles, draw_rack


# Tiles that are nice to keep (cheap, flexible). A light leave heuristic; the
# simulation already handles most of the strategy.
def _leave_value(engine, leave):
    """Small bonus for a balanced, low-point leave (vowel/consonant mix)."""
    if not leave:
        return 0.0
    vowels = sum(1 for t in leave if t in "aeiouyæøå")
    balance = -abs(vowels - len(leave) / 2.0)          # prefer ~half vowels
    dupes = len(leave) - len(set(leave))               # penalise duplicates
    return balance - dupes


class LookaheadAgent:
    def __init__(self, n_samples=8, max_candidates=12, leave_weight=1.0,
                 plies=2, rng=None):
        """``plies``: 2 = my move + opponent reply (defense). 3 = also simulate
        my *next* turn after the opponent (values rack leave directly, since a
        good leave yields a stronger follow-up). 3-ply roughly doubles cost.

        When the bag is empty the opponent's rack is fully known, so evaluation
        switches to a single *exact* rollout instead of sampling (exact endgame
        at the chosen depth)."""
        self.n_samples = n_samples
        self.max_candidates = max_candidates
        self.leave_weight = leave_weight
        self.plies = plies
        self.rng = rng or random.Random()

    # ── core evaluation ──────────────────────────────────────────────────────

    def evaluate(self, engine, board, bonus, rack, max_candidates=None,
                 board_blanks=None):
        """Score every candidate move by lookahead value. Returns a list of
        (value, my_score, exp_opp, move) sorted best-first. Uses only the
        player's own information (board, own rack, unseen-pool distribution).
        ``board_blanks``: positions of blank tiles already on the board."""
        K = max_candidates or self.max_candidates
        board_blanks = set(board_blanks or ())
        moves = engine.legal_moves(board, bonus, rack, board_blanks)[:K]
        if not moves:
            return []

        pool = unseen_tiles(board, rack, board_blanks)
        total = sum(pool.values())
        opp_n = min(7, total)
        endgame = total <= opp_n            # bag empty: pool == opponent's rack
        samples = 1 if endgame else self.n_samples

        results = []
        for m in moves:
            board2 = engine.next_state(board, m)
            # any blank I just placed is a board blank for the opponent's reply
            blanks2 = board_blanks | {(r, c) for r, c, _, b in m.new_tiles if b}
            leave = list(rack)
            for _, _, ch, b in m.new_tiles:
                leave.remove('*' if b else ch)

            opp_acc, my2_acc = 0.0, 0.0
            for _ in range(samples):
                opp_rack = (list(pool.elements()) if endgame
                            else draw_rack(pool, opp_n, self.rng))
                opp_moves = engine.legal_moves(board2, bonus, opp_rack, blanks2)
                opp_best = opp_moves[0] if opp_moves else None
                opp_acc += opp_best.score if opp_best else 0.0

                if self.plies >= 3:         # my next turn after the opponent
                    if opp_best:
                        board3 = engine.next_state(board2, opp_best)
                        blanks3 = blanks2 | {(r, c) for r, c, _, b
                                             in opp_best.new_tiles if b}
                    else:
                        board3, blanks3 = board2, blanks2
                    if endgame:
                        my_next = leave     # bag empty -> no refill
                    else:
                        bag = pool - Counter(opp_rack)
                        my_next = leave + draw_rack(bag, max(0, 7 - len(leave)),
                                                    self.rng)
                    my_moves2 = engine.legal_moves(board3, bonus, my_next, blanks3)
                    my2_acc += my_moves2[0].score if my_moves2 else 0.0

            exp_opp = opp_acc / samples
            value = m.score - exp_opp
            if self.plies >= 3:
                value += my2_acc / samples
            value += self.leave_weight * _leave_value(engine, leave)

            # Endgame: emptying your rack with the bag empty ends the game and
            # awards you the opponent's remaining tile points (known exactly).
            if endgame and not leave:
                value += sum(engine.points.get(t, 0) for t in pool.elements())

            results.append((value, m.score, exp_opp, m))

        results.sort(key=lambda x: -x[0])
        return results

    def pick(self, engine, letters, bonus, rack, board_blanks=None):
        """Best move for a single position (e.g. read from a screenshot).
        Returns (move_or_None, ranked_list)."""
        ranked = self.evaluate(engine, letters, bonus,
                               [t.lower() for t in rack if t],
                               board_blanks=board_blanks)
        return (ranked[0][3] if ranked else None), ranked

    def top_moves(self, engine, letters, bonus, rack, n=10,
                  distinct_words=True, pool_size=None, board_blanks=None):
        """Ranked top-``n`` moves as plain dicts — a fallback list for when the
        official app rejects a word (its dictionary may differ from ours).

        With ``distinct_words`` (default), only the best placement of each word
        is kept: if the app rejects a word, every placement of it is rejected
        too, so the alternatives should be *different words*. Evaluates a larger
        candidate pool so there are enough distinct words to fill the list.
        """
        pool_size = pool_size or max(self.max_candidates, n * 3)
        ranked = self.evaluate(engine, letters, bonus,
                               [t.lower() for t in rack if t],
                               max_candidates=pool_size, board_blanks=board_blanks)
        out, seen = [], set()
        for value, score, exp_opp, m in ranked:
            if distinct_words and m.word in seen:
                continue
            seen.add(m.word)
            out.append({
                "rank": len(out) + 1,
                "word": m.word,
                "direction": m.direction,
                "start": m.start,
                "score": score,
                "value": round(value, 1),
                "exp_opp": round(exp_opp, 1),
                "tiles": [(r, c, ch) for r, c, ch, _ in m.new_tiles],
                "move": m,
            })
            if len(out) >= n:
                break
        return out

    # ── plays a turn of a WordfeudGame (does NOT peek at the opponent's rack) ──

    def act(self, game):
        ranked = self.evaluate(game.engine, game.board, game.bonus,
                               game.rack(), self.max_candidates,
                               board_blanks=game.board_blanks)
        if ranked:
            game.apply_move(ranked[0][3])
        else:
            _fallback(game)


def duel(engine, board_sampler, agent0, agent1, n=20, seed=0, config=None,
         randomize_start=True):
    """Play ``n`` games of agent0 vs agent1 on freshly sampled boards, returning
    stats from agent0's perspective. With ``randomize_start`` the starting seat
    is chosen 50/50 per game (deterministically by seed, so it stays matched
    across conditions) — this cancels the first-move advantage so the win rate
    reflects policy strength, not who moved first."""
    from .game import WordfeudGame
    wins = draws = 0
    margins = []
    for i in range(n):
        rng = random.Random(seed + i)
        bonus = board_sampler(rng)
        g = WordfeudGame(engine, bonus, config, rng=rng)
        # Seat of agent0 (0 = starts first). Independent rng so it doesn't
        # perturb the deck, and identical across conditions sharing seeds.
        a0_seat = 0
        if randomize_start and random.Random((seed + i) * 7 + 1).random() < 0.5:
            a0_seat = 1
        seats = (agent0, agent1) if a0_seat == 0 else (agent1, agent0)
        guard = 0
        while not g.done and guard < 400:
            seats[g.to_move].act(g)
            guard += 1
        wins += (g.winner == a0_seat)
        draws += (g.winner is None)
        margins.append(g.margin(player=a0_seat))
    return {"n": n, "win_rate": wins / n, "draw_rate": draws / n,
            "avg_margin": sum(margins) / n}
