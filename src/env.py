"""Gymnasium environment for Wordfeud (single-agent vs a greedy opponent).

The agent is player 0. Each step it chooses among the current legal placement
moves (candidate-evaluation interface): the env exposes the top-K moves as a
feature matrix plus an action mask, and ``action`` indexes into them (the last
index is "pass/swap"). After the agent acts, the greedy opponent plays its turn
inside ``step`` so the agent sees a clean single-agent MDP.

Reward = the agent's move score each turn, plus the final score margin on the
terminal step.

Note: with K candidates the action space is a fixed ``Discrete(K+1)`` with a
mask — convenient for maskable policies (e.g. sb3-contrib MaskablePPO). The full
``Move`` objects are also returned in ``info["moves"]`` so a custom
candidate-encoding policy can build richer features than the built-in matrix.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .move_generator import BOARD_SIZE, DEFAULT_POINTS
from .game import WordfeudGame, GameConfig
from .agents import GreedyAgent

# letter <-> index (0 = empty; 1..27 = letters, in DEFAULT_POINTS order, no blank)
_LETTERS = [c for c in DEFAULT_POINTS if c != '*']
LETTER_TO_IDX = {c: i + 1 for i, c in enumerate(_LETTERS)}
N_LETTERS = len(_LETTERS)                 # 27
RACK_DIM = N_LETTERS + 1                  # + blank
# per-candidate features: score, n_tiles, is_vertical, start_r, start_c,
# word_len, leave_points (point-sum of tiles kept on the rack after the play)
CAND_FEATURES = 7


def candidate_matrix(engine, rack, moves, K):
    """Encode ``moves`` (≤K) into a (K, CAND_FEATURES) matrix + a (K+1,) action
    mask (last index = pass). Shared by the env and by screenshot inference so a
    policy sees identical features in both."""
    cands = np.zeros((K, CAND_FEATURES), np.float32)
    mask = np.zeros(K + 1, np.int8)
    mask[K] = 1                                  # pass/swap always legal
    rack_pts = sum(engine.points.get(t, 0) for t in rack)
    for i, m in enumerate(moves[:K]):
        r0, c0 = m.start
        used = sum(engine.points.get('*' if b else ch, 0)
                   for _, _, ch, b in m.new_tiles)
        cands[i] = (m.score, len(m.new_tiles), 1.0 if m.direction == 'V' else 0.0,
                    r0, c0, len(m.word), rack_pts - used)
        mask[i] = 1
    return cands, mask


def choose_action(engine, letters, bonus, rack, policy, max_candidates=64,
                  board_blanks=None):
    """Pick a move for a single position (e.g. read from a screenshot) using the
    same candidate features the env feeds during training.

    ``policy(obs, info) -> action_index`` (an index into the candidate list; the
    last index = pass). ``best_candidate_policy`` and ``LinearCandidateAgent.
    act_greedy`` both fit. ``board_blanks``: positions of blank tiles already on
    the board. Returns (move_or_None, moves, action_index).
    """
    rack = [t.lower() for t in rack if t]
    moves = engine.legal_moves(letters, bonus, rack, board_blanks)[:max_candidates]
    cands, mask = candidate_matrix(engine, rack, moves, max_candidates)

    # Full obs so state-aware policies (e.g. the torch agent) work too. Scores
    # and bag are unknown from a single screenshot — use neutral estimates;
    # they are global features and don't change which candidate ranks first much.
    rack_vec = np.zeros(RACK_DIM, np.int8)
    for t in rack:
        rack_vec[N_LETTERS if t == '*' else LETTER_TO_IDX[t] - 1] += 1
    placed = sum(1 for row in letters for ch in row if ch)
    obs = {
        "candidates": cands,
        "action_mask": mask,
        "rack": rack_vec,
        "scores": np.zeros(2, np.float32),
        "bag": np.asarray([max(0, 105 - placed - 14)], np.int16),
    }
    info = {"moves": moves}
    action = policy(obs, info)
    move = moves[action] if action < len(moves) else None     # None = pass
    return move, moves, action


class WordfeudEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, engine, bonus=None, max_candidates=64, opponent=None,
                 config: GameConfig = None, seed=None, board_sampler=None,
                 reward_mode="margin"):
        """``bonus``: a fixed 15x15 layout. ``board_sampler(rng) -> layout``:
        if given, a fresh board is drawn every reset (Wordfeud random mode).

        ``reward_mode``:
          - "margin"  : reward only on the terminal step = final score margin.
                        The signal that correlates with *winning*. Greedy is
                        already per-move optimal, so this is what an agent must
                        learn to *beat* greedy. Sparser, higher variance.
          - "shaped"  : dense per-move score + terminal margin. Easy to learn but
                        pulls the policy toward greedy (own-score maximisation).
        """
        super().__init__()
        self.engine = engine
        self.reward_mode = reward_mode
        self.board_sampler = board_sampler
        if bonus is None and board_sampler is None:
            from .boards import STANDARD_BOARD
            bonus = STANDARD_BOARD
        self.bonus = [list(row) for row in bonus] if bonus is not None else None
        self.K = max_candidates
        self.opponent = opponent or GreedyAgent()
        self.config = config or GameConfig()
        self._rng = np.random.default_rng(seed)
        self.game = None
        self._moves = []

        self.action_space = spaces.Discrete(self.K + 1)   # last index = pass/swap
        self.observation_space = spaces.Dict({
            "board": spaces.Box(0, N_LETTERS, (BOARD_SIZE, BOARD_SIZE), np.int8),
            "bonus": spaces.Box(0, 5, (BOARD_SIZE, BOARD_SIZE), np.int8),
            "rack": spaces.Box(0, 7, (RACK_DIM,), np.int8),
            "scores": spaces.Box(-np.inf, np.inf, (2,), np.float32),
            "bag": spaces.Box(0, 105, (1,), np.int16),
            "candidates": spaces.Box(-np.inf, np.inf, (self.K, CAND_FEATURES), np.float32),
            "action_mask": spaces.Box(0, 1, (self.K + 1,), np.int8),
        })

    # ── gym API ──────────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        import random
        rng = random.Random(int(self._rng.integers(2**31)) if seed is None else seed)
        bonus = self.board_sampler(rng) if self.board_sampler else self.bonus
        self.game = WordfeudGame(self.engine, bonus, self.config, rng=rng)
        self._refresh_moves()
        return self._obs(), self._info()

    def step(self, action):
        g = self.game
        move_score = 0.0

        if action < len(self._moves):
            mv = self._moves[action]
            move_score = float(mv.score)
            g.apply_move(mv)
        else:
            self._fallback()                     # pass/swap (invalid or chosen)

        # Opponent plays until it is the agent's turn again (or game ends).
        while not g.done and g.to_move == 1:
            self.opponent.act(g)

        reward = move_score if self.reward_mode == "shaped" else 0.0
        terminated = g.done
        if terminated:
            reward += float(g.margin(player=0))
        else:
            self._refresh_moves()

        return self._obs(), reward, terminated, False, self._info()

    def render(self):
        return self.game.render()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _fallback(self):
        g = self.game
        rack = g.rack()
        if g.bag_count >= g.cfg.rack_size and rack:
            g.swap(list(rack))
        else:
            g.pass_turn()

    def _refresh_moves(self):
        # Top-K legal placements for the agent (player 0), best score first.
        self._moves = self.game.legal_moves(player=0)[:self.K]

    def _obs(self):
        g = self.game
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), np.int8)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                ch = g.board[r][c]
                if ch:
                    board[r, c] = LETTER_TO_IDX.get(ch, 0)

        rack = np.zeros(RACK_DIM, np.int8)
        for t in g.racks[0]:
            rack[N_LETTERS if t == '*' else LETTER_TO_IDX[t] - 1] += 1

        cands, mask = candidate_matrix(self.engine, g.racks[0], self._moves, self.K)

        return {
            "board": board,
            "bonus": np.asarray(g.bonus, np.int8),
            "rack": rack,
            "scores": np.asarray(g.scores, np.float32),
            "bag": np.asarray([g.bag_count], np.int16),
            "candidates": cands,
            "action_mask": mask,
        }

    def _info(self):
        return {"moves": list(self._moves),
                "action_mask": self._obs_mask()}

    def _obs_mask(self):
        mask = np.zeros(self.K + 1, np.int8)
        mask[self.K] = 1
        mask[:len(self._moves)] = 1
        return mask
