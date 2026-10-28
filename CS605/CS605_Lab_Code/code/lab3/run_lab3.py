"""
CS605 Lab 3 - experiment driver.

Usage:  python lab3/run_lab3.py [--instances 20]
Writes results/lab3/{lab3_runs.csv, lab3_summary.csv, lab3_output.txt, figures}.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lab3.ksat import ALGORITHMS, Evaluator, generate_ksat, is_satisfiable, pretty, to_dimacs  # noqa: E402

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results", "lab3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=20)
    ap.add_argument("--seed", type=int, default=605)
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    # Part B demonstration
    rng = np.random.default_rng(args.seed)
    v, s = generate_ksat(3, 5, 4, rng)
    demo = ["Part B example: k=3, m=5, n=4", "  " + pretty(v, s), to_dimacs(v, s, 4)]

    configs = [(n, r) for n in (20, 50, 100) for r in (3.0, 4.0, 4.3)]
    rows = []
    t_start = time.time()
    for n, ratio in configs:
        m = int(round(ratio * n))
        for inst in range(args.instances):
            irng = np.random.default_rng([args.seed, n, m, inst])
            var, sign = generate_ksat(3, m, n, irng)
            sat = is_satisfiable(var, sign, n)
            for ai, (alg, fn) in enumerate(ALGORITHMS.items()):
                for h in ("h1", "h2"):
                    ev = Evaluator(var, sign, h)
                    rrng = np.random.default_rng([args.seed, n, m, inst, ai])
                    t0 = time.perf_counter()
                    x, ok, L = fn(ev, n, rrng)
                    dt = time.perf_counter() - t0
                    T = ev.generated
                    rows.append(dict(n=n, m=m, ratio=ratio, instance=inst, satisfiable=sat,
                                     algorithm=alg, heuristic=h, solved=bool(ok), L=L, T=T,
                                     penetrance=(L / T if ok and T else 0.0),
                                     final_unsat=m - ev.n_sat(x), seconds=dt))
        print(f"done n={n} m={m}  ({time.time()-t_start:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "lab3_runs.csv"), index=False)
    sat_df = df[df.satisfiable]
    summ = (sat_df.groupby(["n", "m", "algorithm", "heuristic"])
            .agg(success=("solved", "mean"), mean_T=("T", "mean"),
                 mean_L_solved=("L", lambda s: s[sat_df.loc[s.index, "solved"]].mean()),
                 penetrance=("penetrance", "mean"), mean_unsat_left=("final_unsat", "mean"),
                 ms=("seconds", lambda s: 1e3 * s.mean()))
            .reset_index())
    summ.to_csv(os.path.join(RESULTS, "lab3_summary.csv"), index=False)
    satfrac = df.groupby(["n", "m"]).satisfiable.mean().reset_index()
    overall = (sat_df.groupby(["algorithm", "heuristic"])
               .agg(success=("solved", "mean"), penetrance=("penetrance", "mean"), mean_T=("T", "mean"))
               .reset_index())
    with open(os.path.join(RESULTS, "lab3_output.txt"), "w") as f:
        f.write("\n".join(demo) + "\n\n")
        f.write("Fraction of generated instances that are satisfiable (MiniSat):\n")
        f.write(satfrac.to_string(index=False) + "\n\n")
        f.write("Per-configuration results on satisfiable instances:\n")
        f.write(summ.to_string(index=False, float_format=lambda v: f"{v:.4g}") + "\n\n")
        f.write("Overall (all satisfiable instances):\n")
        f.write(overall.to_string(index=False, float_format=lambda v: f"{v:.4g}") + "\n")
    print(open(os.path.join(RESULTS, "lab3_output.txt")).read())


if __name__ == "__main__":
    main()
