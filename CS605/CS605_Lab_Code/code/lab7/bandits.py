"""
CS605 Lab 7 (2)-(4) - Exploration vs. exploitation in n-armed bandits.

(2) binaryBanditA / binaryBanditB: Python ports of the course's MATLAB bandits
    (two actions, Bernoulli rewards with p = [0.1, 0.2] and [0.8, 0.9]).
    Epsilon-greedy with sample-average action values.
(3) bandit_nonstat: 10 arms, all means start equal (0) and take independent
    Gaussian random walks, q*(a) <- q*(a) + N(0, 0.01^2) at every step; the
    reward is N(q*(a), 1).
(4) Modified epsilon-greedy for non-stationary rewards: constant step size
    Q <- Q + alpha (R - Q) (exponential recency-weighted average), compared with
    the sample average Q <- Q + (1/n)(R - Q), over 10 000 steps.
Reference: Sutton & Barto, Reinforcement Learning: An Introduction, 2e, Ch. 2.

All agents are vectorised over independent runs (rows).
Run:  python lab7/bandits.py
"""
import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results", "lab7")


# ----------------------------------------------------------------------------
# Environments
# ----------------------------------------------------------------------------
class BinaryBandit:
    """binaryBanditA: p = [0.1, 0.2];  binaryBanditB: p = [0.8, 0.9] (actions 1, 2 -> index 0, 1)."""

    def __init__(self, p, runs, rng):
        self.p = np.asarray(p)
        self.runs, self.rng = runs, rng
        self.k = len(p)

    def pull(self, a):                        # a: (runs,) action indices
        return (self.rng.random(self.runs) < self.p[a]).astype(float)

    def optimal(self):
        return np.full(self.runs, int(np.argmax(self.p)))


def binaryBanditA(action, rng=np.random.default_rng()):
    """Direct port of binaryBanditA.m for a single action in {1, 2}."""
    return 1 if rng.random() < [0.1, 0.2][action - 1] else 0


def binaryBanditB(action, rng=np.random.default_rng()):
    """Direct port of binaryBanditB.m for a single action in {1, 2}."""
    return 1 if rng.random() < [0.8, 0.9][action - 1] else 0


class NonStationaryBandit:
    """bandit_nonstat: 10 arms, equal initial means, independent random walks (sd 0.01)."""

    def __init__(self, runs, rng, k=10, walk_sd=0.01):
        self.k, self.runs, self.rng, self.sd = k, runs, rng, walk_sd
        self.q = np.zeros((runs, k))

    def pull(self, a):
        r = self.q[np.arange(self.runs), a] + self.rng.standard_normal(self.runs)
        self.q += self.sd * self.rng.standard_normal(self.q.shape)   # all means drift every step
        return r

    def optimal(self):
        return np.argmax(self.q, axis=1)


# ----------------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------------
def epsilon_greedy(env, steps, eps, rng, alpha=None, q0=0.0):
    runs, k = env.runs, env.k
    Q = np.full((runs, k), q0, float)
    N = np.zeros((runs, k))
    rew, opt = np.zeros(steps), np.zeros(steps)
    idx = np.arange(runs)
    for t in range(steps):
        greedy = np.argmax(Q + 1e-9 * rng.random((runs, k)), axis=1)     # random tie-break
        explore = rng.random(runs) < eps
        a = np.where(explore, rng.integers(0, k, runs), greedy)
        best = env.optimal()
        r = env.pull(a)
        N[idx, a] += 1
        step = (1.0 / N[idx, a]) if alpha is None else alpha
        Q[idx, a] += step * (r - Q[idx, a])
        rew[t] = r.mean()
        opt[t] = np.mean(a == best)
    return rew, opt, Q


def smooth(x, w=50):
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main():
    os.makedirs(RESULTS, exist_ok=True)
    summary = {}

    # ---------------- (2) binary bandits ----------------
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    steps, runs = 1000, 2000
    for name, p in (("A", [0.1, 0.2]), ("B", [0.8, 0.9])):
        summary[f"binary_{name}"] = {}
        for eps, col in ((0.0, "#777"), (0.01, "#e9a03b"), (0.1, "#2a9d8f"), (0.2, "#d1495b")):
            rng = np.random.default_rng([7, int(eps * 100), ord(name)])
            env = BinaryBandit(p, runs, rng)
            rew, opt, Q = epsilon_greedy(env, steps, eps, rng)
            summary[f"binary_{name}"][str(eps)] = dict(
                mean_reward_last200=float(rew[-200:].mean()), pct_optimal_last200=float(opt[-200:].mean()),
                mean_Q=Q.mean(0).tolist())
            ax[0 if name == "A" else 1].plot(smooth(opt, 20), color=col, lw=1, label=f"ε = {eps}")
        a = ax[0 if name == "A" else 1]
        a.set_title(f"binaryBandit{name}  p = {p}", fontsize=9)
        a.set_xlabel("step"); a.set_ylabel("% optimal action"); a.grid(alpha=0.3); a.set_ylim(0, 1)
    ax[0].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "binary_bandits.png"), dpi=150); plt.close()

    # ---------------- (3)+(4) non-stationary 10-armed bandit ----------------
    steps, runs = 10000, 1000
    configs = [("sample average, ε=0.1", dict(eps=0.1, alpha=None), "#777"),
               ("constant α=0.1, ε=0.1", dict(eps=0.1, alpha=0.1), "#2a9d8f"),
               ("constant α=0.05, ε=0.1", dict(eps=0.1, alpha=0.05), "#e9a03b"),
               ("constant α=0.1, ε=0.05", dict(eps=0.05, alpha=0.1), "#d1495b")]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
    summary["nonstationary"] = {}
    for label, cfg, col in configs:
        rng = np.random.default_rng(42)          # same environment sequence for every agent
        env = NonStationaryBandit(runs, rng)
        rew, opt, _ = epsilon_greedy(env, steps, cfg["eps"], np.random.default_rng(43), alpha=cfg["alpha"])
        summary["nonstationary"][label] = dict(
            reward_last1000=float(rew[-1000:].mean()), pct_opt_last1000=float(opt[-1000:].mean()),
            reward_first1000=float(rew[:1000].mean()), pct_opt_first1000=float(opt[:1000].mean()))
        ax[0].plot(smooth(rew, 200), color=col, lw=1, label=label)
        ax[1].plot(smooth(opt, 200), color=col, lw=1, label=label)
    ax[0].set_ylabel("average reward"); ax[1].set_ylabel("% optimal action")
    for a in ax:
        a.set_xlabel("step"); a.grid(alpha=0.3)
    ax[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "nonstationary_bandit.png"), dpi=150); plt.close()

    # how far the true means drift (illustration)
    rng = np.random.default_rng(1)
    env = NonStationaryBandit(1, rng)
    traj = []
    for _ in range(10000):
        env.pull(np.array([0]))
        traj.append(env.q[0].copy())
    traj = np.array(traj)
    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.plot(traj, lw=0.7)
    ax.set_xlabel("step"); ax.set_ylabel("q*(a)"); ax.set_title("bandit_nonstat: true means of the 10 arms", fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "nonstat_true_means.png"), dpi=150); plt.close()
    summary["drift_example_final_spread"] = float(traj[-1].max() - traj[-1].min())
    summary["optimal_arm_changes_example"] = int(np.sum(np.diff(np.argmax(traj, axis=1)) != 0))
    print(json.dumps(summary, indent=1))
    json.dump(summary, open(os.path.join(RESULTS, "lab7_bandit_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
