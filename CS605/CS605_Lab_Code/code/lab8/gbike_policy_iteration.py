"""
CS605 Lab 8 (2)-(3) - Gbike bicycle rental: a continuing finite MDP solved by policy iteration.

MDP formulation (problem 2)
  time step : one day;  discount gamma = 0.9
  state     : s = (n1, n2), bikes at each location at the end of the day, 0..20 each (441 states)
  action    : a in {-5..5}, net bikes moved overnight from location 1 to 2 (a < 0: 2 -> 1),
              feasible if a <= n1, -a <= n2 and neither location exceeds 20 after the move
  dynamics  : morning stock m_i = n_i - a (loc 1), n_2 + a (loc 2);
              requests  Q_i ~ Poisson(3), Poisson(4);  rentals = min(Q_i, m_i)
              returns   B_i ~ Poisson(3), Poisson(2), usable from the next day;
              next state n_i' = min(m_i - rentals_i + B_i, 20)
  reward    : 10 * (rentals_1 + rentals_2)  -  2 * |a|
Because the two locations evolve independently given the action, the expected
next-state value is  E[V(s')] = P1[m1, :] V P2[m2, :]^T, computed with matrix products.

Problem 3 (modified):  the first bike moved 1 -> 2 each night is free; every other bike
costs INR 2; and if more than 10 bikes are kept overnight at a location (after moving)
an extra INR 4 is paid for that location.

Run:  python lab8/gbike_policy_iteration.py
"""
import json
import os

import matplotlib
import numpy as np
from scipy.stats import poisson

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results", "lab8")
MAXB, MAXMOVE, GAMMA = 20, 5, 0.9
REQ, RET = (3, 4), (3, 2)
RENT, MOVE_COST = 10.0, 2.0
ACTIONS = np.arange(-MAXMOVE, MAXMOVE + 1)


def location_model(lam_req, lam_ret):
    """P[m, n'] = P(next end-of-day count n' | morning stock m); R[m] = expected rental income."""
    P = np.zeros((MAXB + 1, MAXB + 1))
    R = np.zeros(MAXB + 1)
    for m in range(MAXB + 1):
        req = poisson.pmf(np.arange(m), lam_req)
        req = np.append(req, 1 - req.sum())               # P(rentals = m) = P(Q >= m)
        R[m] = RENT * np.dot(np.arange(m + 1), req)
        for rented, pr in enumerate(req):
            left = m - rented
            ret = poisson.pmf(np.arange(MAXB - left), lam_ret)
            ret = np.append(ret, 1 - ret.sum())             # everything above capacity is lost
            P[m, left:] += pr * ret
    return P, R


P1, R1 = location_model(REQ[0], RET[0])
P2, R2 = location_model(REQ[1], RET[1])


def feasible(n1, n2, a):
    return (a <= n1) and (-a <= n2) and (n1 - a <= MAXB) and (n2 + a <= MAXB)


def action_cost(a, m1, m2, modified):
    if not modified:
        return MOVE_COST * abs(a)
    cost = MOVE_COST * (a - 1) if a > 0 else MOVE_COST * abs(a)    # one free shuttle 1 -> 2
    cost += 4.0 * (m1 > 10) + 4.0 * (m2 > 10)                       # second parking lot
    return cost


def q_value(n1, n2, a, V, modified):
    m1, m2 = n1 - a, n2 + a
    ev = P1[m1] @ V @ P2[m2]
    return R1[m1] + R2[m2] - action_cost(a, m1, m2, modified) + GAMMA * ev


def policy_evaluation(pi, V, modified, theta=1e-4):
    sweeps = 0
    while True:
        delta = 0.0
        for n1 in range(MAXB + 1):
            for n2 in range(MAXB + 1):
                v = q_value(n1, n2, pi[n1, n2], V, modified)      # in-place (Gauss-Seidel) update
                delta = max(delta, abs(v - V[n1, n2]))
                V[n1, n2] = v
        sweeps += 1
        if delta < theta:
            return V, sweeps


def policy_improvement(pi, V, modified):
    stable = True
    new = pi.copy()
    for n1 in range(MAXB + 1):
        for n2 in range(MAXB + 1):
            best_a, best_q = pi[n1, n2], q_value(n1, n2, pi[n1, n2], V, modified)
            for a in ACTIONS:
                if feasible(n1, n2, a):
                    q = q_value(n1, n2, a, V, modified)
                    if q > best_q + 1e-9:
                        best_a, best_q = a, q
            new[n1, n2] = best_a
            if best_a != pi[n1, n2]:
                stable = False
    return new, stable


def policy_iteration(modified):
    pi = np.zeros((MAXB + 1, MAXB + 1), dtype=int)
    V = np.zeros((MAXB + 1, MAXB + 1))
    history = [pi.copy()]
    evals = []
    while True:
        V, sweeps = policy_evaluation(pi, V, modified)
        evals.append(sweeps)
        pi, stable = policy_improvement(pi, V, modified)
        history.append(pi.copy())
        print(f"  {'modified' if modified else 'original'}: iteration {len(evals)}, "
              f"evaluation sweeps {sweeps}, V(10,10) = {V[10, 10]:.2f}, stable = {stable}", flush=True)
        if stable:
            return pi, V, history, evals


def plot_policies(history, title, path):
    k = len(history)
    fig, axes = plt.subplots(1, k, figsize=(2.3 * k, 2.6))
    for i, (ax, pi) in enumerate(zip(np.atleast_1d(axes), history)):
        im = ax.imshow(pi, origin="lower", cmap="RdBu_r", vmin=-5, vmax=5)
        ax.contour(pi, levels=np.arange(-5.5, 6), colors="k", linewidths=0.3)
        ax.set_title(f"π{i}", fontsize=9)
        ax.set_xlabel("# bikes at loc 2", fontsize=7); ax.set_ylabel("# bikes at loc 1", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.colorbar(im, ax=axes, fraction=0.02, label="bikes moved 1→2")
    fig.suptitle(title, fontsize=9)
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def plot_value(V, title, path):
    fig = plt.figure(figsize=(4.2, 3.4))
    ax = fig.add_subplot(projection="3d")
    X, Y = np.meshgrid(np.arange(MAXB + 1), np.arange(MAXB + 1))
    ax.plot_surface(X, Y, V, cmap="viridis", lw=0)
    ax.set_xlabel("loc 2", fontsize=7); ax.set_ylabel("loc 1", fontsize=7); ax.set_zlabel("v(s)", fontsize=7)
    ax.tick_params(labelsize=6); ax.set_title(title, fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main():
    os.makedirs(RESULTS, exist_ok=True)
    summary = {}
    for modified, tag in ((False, "original"), (True, "modified")):
        pi, V, hist, evals = policy_iteration(modified)
        plot_policies(hist, f"Gbike policy iteration ({tag} problem)", os.path.join(RESULTS, f"gbike_policies_{tag}.png"))
        plot_value(V, f"v_π* ({tag})", os.path.join(RESULTS, f"gbike_value_{tag}.png"))
        np.savetxt(os.path.join(RESULTS, f"gbike_policy_{tag}.csv"), pi, fmt="%d", delimiter=",")
        summary[tag] = dict(iterations=len(evals), eval_sweeps=evals, V_min=float(V.min()), V_max=float(V.max()),
                            V_10_10=float(V[10, 10]), V_20_20=float(V[20, 20]), V_0_0=float(V[0, 0]),
                            policy=pi.tolist(),
                            moved_1to2_states=int((pi > 0).sum()), moved_2to1_states=int((pi < 0).sum()))
    a, b = np.array(summary["original"]["policy"]), np.array(summary["modified"]["policy"])
    summary["states_where_policy_differs"] = int((a != b).sum())
    json.dump(summary, open(os.path.join(RESULTS, "lab8_gbike_summary.json"), "w"), indent=2)
    print({k: {kk: vv for kk, vv in v.items() if kk != "policy"} if isinstance(v, dict) else v
           for k, v in summary.items()})


if __name__ == "__main__":
    main()
