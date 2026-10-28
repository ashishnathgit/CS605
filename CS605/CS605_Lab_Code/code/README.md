# CS605 Artificial Intelligence — Lab code (Autumn 2026-27)

Python 3.10+ implementations of the eight lab assignments in the CS605 lab manual
(IIIT Vadodara). Every script writes its numbers and figures to `../results/labN/`,
and those are the numbers quoted in the two reports.

```
pip install -r requirements.txt
python run_all.py            # everything (~6 min on a laptop; Lab 3 is the slowest)
python lab1/lab1_bfs_dfs.py  # or any single lab, always run from this code/ folder
```

| Lab | Script(s) | What it does |
|---|---|---|
| common | `common/search.py` | One graph-search agent (frontier + explored hash table) giving BFS, DFS, UCS, greedy best-first and A* — reused by Labs 1–2 |
| 1 | `lab1/lab1_bfs_dfs.py` | Missionaries & Cannibals and Rabbit Leap: state-space size, BFS/DFS solutions, optimality check, time/memory, scaling with k |
| 2 | `lab2/plagiarism_astar.py`, `lab2/test_docs.py` | Sentence alignment by A* with Levenshtein cost; three admissible heuristics; the four required test cases plus an insertion/deletion case; scaling study |
| 3 | `lab3/ksat.py`, `lab3/run_lab3.py`, `lab3/plot_lab3.py` | Uniform random k-SAT generator; Hill-Climbing, Beam search (w = 3, 4), VND (3 neighbourhoods); heuristics h1/h2 compared on success and penetrance; MiniSat used only to label satisfiable instances |
| 4 | `lab4/jigsaw_sa.py` | Scrambled-Lena jigsaw by simulated annealing (two seam energies, block moves, ablations) |
| 4 (in-lab) | `lab4/tsp_rajasthan_sa.py` | 20-place Rajasthan tour by SA with 2-opt moves, checked against the exact Held–Karp optimum |
| 4 (challenge) | `lab4/raag_bhairav_sa.py` | Raag Bhairav melody by SA with a grammar-based energy; writes `bhairav_sa.wav` |
| 5 | `lab5/hmm_finance.py` | Gaussian HMM regimes for S&P 500 and NASDAQ daily returns, K = 2, 3, 4, transition matrix, regime forecast |
| 6 | `lab6/hopfield.py` | 10×10 associative memory, capacity, error correction, eight-rook network, 10-city Hopfield–Tank TSP |
| 7 | `lab7/menace.py`, `lab7/bandits.py` | MENACE in Python; binaryBanditA/B with ε-greedy; non-stationary 10-armed bandit and the constant-step-size agent |
| 8 | `lab8/gridworld_vi.py`, `lab8/gbike_policy_iteration.py` | Value iteration on the 4×3 world for several r(s); Gbike policy iteration, original and modified problems |

## Data
* `data/scrambled_lena.mat` — from the course repository (`pratikiiitv/CS605`). It is an Octave
  *text* file, so `jigsaw_sa.py` parses it directly (column-major order).
* Lab 5 tries Yahoo Finance first (`yfinance`). If Yahoo is unreachable it falls back to the
  Yahoo-sourced daily S&P 500 / NASDAQ series shipped with the `arch` package (1999–2018);
  the report results were produced with that fallback. Run
  `python lab5/hmm_finance.py --start 2010-01-01 --end 2026-10-27` on a machine with internet
  access to refit on current data.
* `binaryBanditA/B` and the Gbike reference code come from the instructor's earlier repository
  (`pratikiiitv/CS307`); they are re-implemented here in Python.

## Reproducibility
All random generators are seeded; rerunning a script reproduces the reported numbers.
