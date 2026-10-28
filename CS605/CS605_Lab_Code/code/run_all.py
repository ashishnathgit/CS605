"""Run every lab script in order (from the code/ directory)."""
import subprocess
import sys

SCRIPTS = ["lab1/lab1_bfs_dfs.py", "lab2/plagiarism_astar.py", "common/plot_labs12.py",
           "lab3/run_lab3.py", "lab3/plot_lab3.py", "lab4/jigsaw_sa.py", "lab4/tsp_rajasthan_sa.py",
           "lab4/raag_bhairav_sa.py", "lab5/hmm_finance.py", "lab6/hopfield.py", "lab7/menace.py",
           "lab7/bandits.py", "lab8/gridworld_vi.py", "lab8/gbike_policy_iteration.py"]

for s in SCRIPTS:
    print(f"\n######## {s}", flush=True)
    subprocess.run([sys.executable, s], check=True)
