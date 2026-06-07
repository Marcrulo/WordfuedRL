"""Evaluation harness: play an agent against the env's greedy opponent."""

from __future__ import annotations


def best_candidate_policy(obs, info):
    """Baseline policy: always play the highest-scoring legal move (= greedy)."""
    if info["moves"]:
        return 0
    return obs["action_mask"].shape[0] - 1        # pass index (K)


def play_episode(env, policy, seed=None):
    """Run one episode; return (winner, agent_margin, summed_reward)."""
    obs, info = env.reset(seed=seed)
    done = False
    total = 0.0
    while not done:
        obs, r, done, _, info = env.step(policy(obs, info))
        total += r
    return env.game.winner, env.game.margin(player=0), total


def evaluate(env, policy, n=50, seed0=0):
    """Aggregate stats for ``policy`` (agent = player 0) over ``n`` games."""
    wins = draws = 0
    margins = []
    for i in range(n):
        winner, margin, _ = play_episode(env, policy, seed=seed0 + i)
        wins += (winner == 0)
        draws += (winner is None)
        margins.append(margin)
    return {
        "n": n,
        "win_rate": wins / n,
        "draw_rate": draws / n,
        "avg_margin": sum(margins) / n,
    }
