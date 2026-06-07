"""A non-greedy learned agent: PyTorch actor-critic over candidate moves.

Improves on the linear REINFORCE starter in two ways:
  1. Richer features per candidate — not just score, but the *rack leave*
     composition (vowels/consonants/blanks/duplicates) and global game state
     (score margin, tiles left). These encode what greedy ignores.
  2. A value-function critic as the baseline, which sharply cuts the gradient
     variance that left the linear agent stuck at greedy.

The policy scores each legal candidate, masks the rest, and samples (train) or
argmaxes (play). Trained with batched advantage actor-critic against the env's
greedy opponent on randomised boards.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import LETTER_TO_IDX, N_LETTERS, CAND_FEATURES

_VOWELS = set("aeiouyæøå")
_VOWEL_IDX = np.array([LETTER_TO_IDX[c] - 1 for c in _VOWELS], dtype=np.int64)

GLOBAL_FEATURES = 4                                   # score_diff, bag, rack_pts, n_moves
FEATURE_DIM = CAND_FEATURES + 5 + GLOBAL_FEATURES     # 7 + leave-comp(5) + globals(4)


def _featurize(obs, info):
    """Return (X, valid) where X is (k, FEATURE_DIM) float32 for the k legal
    moves (k may be 0). Pure numpy; consumed by the net as a tensor."""
    moves = info["moves"]
    k = len(moves)
    if k == 0:
        return np.zeros((0, FEATURE_DIM), np.float32)

    base = obs["candidates"][:k].astype(np.float32)   # score..leave_pts (7)
    rack = obs["rack"].astype(np.float32)             # 28 counts
    scores = obs["scores"]
    bag = float(obs["bag"][0])

    # Global features (shared across candidates), roughly normalised.
    g = np.array([(scores[0] - scores[1]) / 100.0, bag / 100.0,
                  rack[:N_LETTERS].sum() / 7.0, k / 64.0], np.float32)

    feats = np.zeros((k, FEATURE_DIM), np.float32)
    for i, m in enumerate(moves):
        # rack leave = rack minus the tiles this move spends
        leave = rack.copy()
        for _, _, ch, blank in m.new_tiles:
            leave[N_LETTERS if blank else LETTER_TO_IDX[ch] - 1] -= 1
        letters = leave[:N_LETTERS]
        vowels = letters[_VOWEL_IDX].sum()
        comp = np.array([
            vowels / 7.0,
            (letters.sum() - vowels) / 7.0,            # consonants
            leave[N_LETTERS] / 2.0,                    # blanks kept
            (letters > 0).sum() / 7.0,                 # distinct letters
            np.maximum(letters - 1, 0).sum() / 7.0,    # duplicates
        ], np.float32)
        # normalise the base features to comparable scales
        s, nt, vert, r0, c0, wl, lp = base[i]
        nb = np.array([s / 50.0, nt / 7.0, vert, r0 / 14.0, c0 / 14.0,
                       wl / 15.0, lp / 40.0], np.float32)
        feats[i] = np.concatenate([nb, comp, g])
    return feats


class ActorCritic(nn.Module):
    def __init__(self, dim=FEATURE_DIM, hidden=64):
        super().__init__()
        # Actor trunk: per-candidate scorer.
        self.body = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        self.actor = nn.Linear(hidden, 1)
        # Critic is a *separate* network on a permutation-invariant summary of the
        # raw features (mean+max pool + globals). Keeping it off the actor trunk
        # stops its large value gradients from corrupting the policy.
        self.critic = nn.Sequential(nn.Linear(2 * dim + GLOBAL_FEATURES, hidden),
                                    nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                    nn.Linear(hidden, 1))

    def forward(self, X, g):
        logits = self.actor(self.body(X)).squeeze(-1)  # (k,)
        summary = torch.cat([X.mean(0), X.amax(0), g])
        value = self.critic(summary).squeeze(-1)       # scalar (scaled units)
        return logits, value


class TorchCandidateAgent:
    def __init__(self, device="cpu", seed=0):
        torch.manual_seed(seed)
        self.device = device
        self.net = ActorCritic().to(device)

    def _tensors(self, obs, info):
        X = _featurize(obs, info)
        if X.shape[0] == 0:
            return None, None
        g = X[0, -GLOBAL_FEATURES:]                    # globals are identical per row
        return (torch.from_numpy(X).to(self.device),
                torch.from_numpy(g).to(self.device))

    @torch.no_grad()
    def act_greedy(self, obs, info):
        X, g = self._tensors(obs, info)
        if X is None:
            return obs["action_mask"].shape[0] - 1     # pass
        logits, _ = self.net(X, g)
        return int(torch.argmax(logits).item())

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    @classmethod
    def load(cls, path, device="cpu"):
        agent = cls(device=device)
        agent.net.load_state_dict(torch.load(path, map_location=device))
        agent.net.eval()
        return agent


def train(env, agent=None, batches=60, episodes_per_batch=8, lr=1e-3,
          gamma=0.999, entropy_coef=0.01, value_coef=0.5, reward_scale=100.0,
          seed=0, log_every=5):
    """Batched advantage actor-critic against the env's greedy opponent.
    Returns (agent, history of mean batch return). ``reward_scale`` divides
    returns so the value target is O(1) — keeps the critic well-conditioned."""
    rng = np.random.default_rng(seed)
    agent = agent or TorchCandidateAgent(seed=seed)
    opt = torch.optim.Adam(agent.net.parameters(), lr=lr)
    history = []

    for b in range(batches):
        logps, entropies, values, rets = [], [], [], []
        batch_returns = []

        for _ in range(episodes_per_batch):
            obs, info = env.reset(seed=int(rng.integers(2**31)))
            ep_logp, ep_ent, ep_val, rewards, decided = [], [], [], [], []
            done = False
            while not done:
                X, g = agent._tensors(obs, info)
                if X is None:                          # forced pass: no decision
                    a = obs["action_mask"].shape[0] - 1
                    obs, r, done, _, info = env.step(a)
                    rewards.append(r); decided.append(False)
                    continue
                logits, value = agent.net(X, g)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                ep_logp.append(dist.log_prob(a))
                ep_ent.append(dist.entropy())
                ep_val.append(value)
                obs, r, done, _, info = env.step(int(a.item()))
                rewards.append(r); decided.append(True)

            # discounted returns-to-go for every step; keep decision steps only
            G = np.zeros(len(rewards), np.float32)
            acc = 0.0
            for t in reversed(range(len(rewards))):
                acc = rewards[t] + gamma * acc
                G[t] = acc
            for t, was_decision in enumerate(decided):
                if was_decision:
                    rets.append(G[t] / reward_scale)   # scaled value target
            logps += ep_logp; entropies += ep_ent; values += ep_val
            batch_returns.append(G[0] if len(G) else 0.0)

        if not logps:                                  # degenerate batch
            continue
        logp = torch.stack(logps)
        ent = torch.stack(entropies)
        val = torch.stack(values)
        ret = torch.tensor(rets, dtype=torch.float32, device=agent.device)
        adv = ret - val.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        actor_loss = -(adv * logp).mean()
        critic_loss = F.mse_loss(val, ret)
        loss = actor_loss + value_coef * critic_loss - entropy_coef * ent.mean()

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.net.parameters(), 1.0)
        opt.step()

        history.append(float(np.mean(batch_returns)))
        if log_every and (b + 1) % log_every == 0:
            print(f"batch {b+1:3d}/{batches}  mean return: {history[-1]:8.1f}  "
                  f"actor {actor_loss.item():.3f}  critic {critic_loss.item():.1f}")
    return agent, history
