"""A minimal trainable candidate-evaluation agent (numpy REINFORCE).

This is a starter, not a strong player: a linear policy that scores each
candidate move from its feature vector and samples one. It exists to prove the
candidate-evaluation RL loop end-to-end without extra dependencies. The obvious
upgrade is to swap the linear scorer for a neural net (torch) that also consumes
the board/rack planes — the env already exposes everything needed.

Policy:  logits = standardised(candidate_features) @ w ;  softmax over the
legal candidates;  sample.  Trained with REINFORCE (return-to-go + baseline).
"""

from __future__ import annotations

import numpy as np


class LinearCandidateAgent:
    def __init__(self, n_features, seed=0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 0.01, size=n_features)

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path):
        np.save(path, self.w)

    @classmethod
    def load(cls, path):
        w = np.load(path)
        agent = cls(len(w))
        agent.w = w
        return agent

    # ── feature prep ─────────────────────────────────────────────────────────

    @staticmethod
    def _valid_features(obs, info):
        """Standardised feature matrix for the legal candidates (k, F)."""
        k = len(info["moves"])
        X = obs["candidates"][:k].astype(np.float64)
        if k > 1:
            mu, sd = X.mean(0), X.std(0)
            sd[sd == 0] = 1.0
            X = (X - mu) / sd
        return X

    def _probs(self, X):
        logits = X @ self.w
        logits -= logits.max()
        e = np.exp(logits)
        return e / e.sum()

    # ── acting ───────────────────────────────────────────────────────────────

    def act_greedy(self, obs, info):
        """Deterministic policy for evaluation: pick the argmax candidate."""
        if not info["moves"]:
            return obs["action_mask"].shape[0] - 1
        return int(np.argmax(self._valid_features(obs, info) @ self.w))

    def act_sample(self, obs, info):
        """Stochastic policy for training; also returns (X, probs, action)."""
        if not info["moves"]:
            return obs["action_mask"].shape[0] - 1, None, None
        X = self._valid_features(obs, info)
        p = self._probs(X)
        a = int(np.random.choice(len(p), p=p))
        return a, X, p


def _run_episode(env, agent, seed):
    """Play one training episode; return (grads, returns-to-go, episode_return)."""
    obs, info = env.reset(seed=seed)
    grads, rewards = [], []
    done = False
    while not done:
        a, X, p = agent.act_sample(obs, info)
        grad = np.zeros_like(agent.w) if X is None else (X[a] - p @ X)
        obs, r, done, _, info = env.step(a)
        grads.append(grad)
        rewards.append(r)
    return grads, rewards


def train(env, episodes=400, batch=16, lr=0.05, gamma=0.999, seed=0, log_every=4):
    """Batched REINFORCE. The baseline is computed *across* the batch, so the
    signal "this episode beat the others" is preserved. Returns (agent, history
    of mean batch return)."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    from .env import CAND_FEATURES
    agent = LinearCandidateAgent(CAND_FEATURES, seed=seed)
    history = []

    n_batches = max(1, episodes // batch)
    for b in range(n_batches):
        batch_grads, batch_adv, ep_returns = [], [], []
        for _ in range(batch):
            grads, rewards = _run_episode(env, agent, int(rng.integers(2**31)))
            G, acc = [0.0] * len(rewards), 0.0
            for t in reversed(range(len(rewards))):
                acc = rewards[t] + gamma * acc
                G[t] = acc
            batch_grads.extend(grads)
            batch_adv.extend(G)
            ep_returns.append(G[0] if G else 0.0)

        adv = np.asarray(batch_adv)
        adv -= adv.mean()                          # baseline across the whole batch
        if adv.std() > 1e-8:
            adv /= adv.std()
        update = sum(a_t * g_t for a_t, g_t in zip(adv, batch_grads))
        agent.w += lr * update / len(batch_grads)

        history.append(np.mean(ep_returns))
        if log_every and (b + 1) % log_every == 0:
            print(f"batch {b+1:3d}/{n_batches}  mean return: {history[-1]:8.1f}")
    return agent, history
