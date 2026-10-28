"""
CS605 Lab 4 - Challenge problem: a melody in Raag Bhairav by simulated annealing.

Grammar of Bhairav used here (Bhairav thaat, morning sandhiprakash raag):
  swaras   : S r(komal) G m P d(komal) N, with mandra P. d. N. and taar S'
  aaroh    : S r G m P d N S'          avroh : S' N d P m G r S
  pakad    : G m d d P , G m r r S      vadi d (komal Dha), samvadi r (komal Re)
  nyaas    : S, r, m, P, d - phrases come to rest on these notes
Representation: a sequence of 32 swaras = 4 phrases of 8 matras (two cycles of
16-beat teentaal).  Only Bhairav swaras can be represented, so 'no foreign
note' holds by construction.

Energy (to be minimised) = weighted sum of penalties
  E1  melodic leaps larger than 5 semitones (except S<->P / P<->S' consonances)
  E2  ascending movement that skips the aaroh order badly / descending movement
      against the avroh (moves outside the raag's chalan)
  E3  phrase endings that are not nyaas swaras; melody must end on S
  E4  missing the vadi/samvadi emphasis (target shares for d and r)
  E5  missing characteristic phrases (pakad fragments) - reward each found
  E6  monotony: the same swara repeated more than twice in a row
SA moves: change one swara, shift a swara by one step in the scale, or copy a
random window of the current melody to another position (motif repetition).

Run:  python lab4/raag_bhairav_sa.py       -> sargam text + WAV file in results/lab4
"""
import json
import os

import numpy as np
from scipy.io import wavfile

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results", "lab4")

# swara name -> semitones above madhya Sa
SCALE = [("P.", -5), ("d.", -4), ("N.", -1), ("S", 0), ("r", 1), ("G", 4), ("m", 5),
         ("P", 7), ("d", 8), ("N", 11), ("S'", 12)]
NAMES = [s for s, _ in SCALE]
SEMI = np.array([v for _, v in SCALE])
IDX = {s: i for i, s in enumerate(NAMES)}
NYAAS = {IDX[s] for s in ("S", "r", "m", "P", "d", "S'")}
PHRASES = {                                   # characteristic fragments (pakad / chalan)
    "G m d d P": 3.0, "G m r r S": 3.0, "d N S'": 1.5, "S' N d P": 2.0,
    "m G r S": 2.0, "S r G m": 1.0, "d P m G": 1.0, "N. S r S": 1.0,
}
PHRASES = {tuple(IDX[x] for x in k.split()): w for k, w in PHRASES.items()}
LENGTH, PHRASE_LEN = 32, 8


def contains(seq, pat):
    n, k = len(seq), len(pat)
    return any(tuple(seq[i:i + k]) == pat for i in range(n - k + 1))


def energy(mel, detail=False):
    mel = list(mel)
    st = SEMI[mel]
    steps = np.diff(st)
    e = {}
    big = np.abs(steps) > 5
    consonant = np.isin(np.abs(steps), [7, 12]) & (np.isin(st[:-1] % 12, [0, 7]))
    e["leaps"] = 2.0 * np.sum(big & ~consonant)
    # chalan: ascending moves should go to the next or next-but-one aaroh swara,
    # descending moves likewise along the avroh (r and P may be skipped in aaroh)
    pos = np.array(mel)
    up_skip = np.sum((np.diff(pos) > 2))
    down_skip = np.sum((np.diff(pos) < -2))
    e["chalan"] = 1.0 * (up_skip + down_skip)
    ends = [mel[i + PHRASE_LEN - 1] for i in range(0, LENGTH, PHRASE_LEN)]
    e["nyaas"] = 2.0 * sum(x not in NYAAS for x in ends) + (0 if mel[-1] == IDX["S"] else 5.0)
    e["start"] = 0 if mel[0] in (IDX["S"], IDX["G"], IDX["N."]) else 1.0
    share_d = np.mean([x in (IDX["d"], IDX["d."]) for x in mel])
    share_r = np.mean([x == IDX["r"] for x in mel])
    e["vadi_samvadi"] = 10 * (max(0, 0.18 - share_d) + max(0, 0.14 - share_r))
    found = [p for p in PHRASES if contains(mel, p)]
    e["phrases"] = sum(PHRASES.values()) - sum(PHRASES[p] for p in found)
    runs = 0
    for i in range(len(mel) - 2):
        if mel[i] == mel[i + 1] == mel[i + 2]:
            runs += 1
    e["monotony"] = 1.5 * runs
    total = float(sum(e.values()))
    if detail:
        return total, e, [" ".join(NAMES[i] for i in p) for p in found]
    return total


def neighbour(mel, rng):
    m = mel.copy()
    r = rng.random()
    i = rng.integers(LENGTH)
    if r < 0.45:
        m[i] = rng.integers(len(NAMES))
    elif r < 0.8:
        m[i] = np.clip(m[i] + rng.choice([-1, 1]), 0, len(NAMES) - 1)
    else:
        k = rng.integers(3, 6)
        a, b = rng.integers(0, LENGTH - k, 2)
        m[b:b + k] = mel[a:a + k]
    return m


def anneal(rng, iters=40000, T0=3.0, Tend=0.01):
    mel = rng.integers(len(NAMES), size=LENGTH)
    E = energy(mel)
    alpha = (Tend / T0) ** (1 / iters)
    T, best, bestE, trace = T0, mel.copy(), E, []
    for it in range(iters):
        cand = neighbour(mel, rng)
        Ec = energy(cand)
        if Ec <= E or rng.random() < np.exp(-(Ec - E) / T):
            mel, E = cand, Ec
            if E < bestE:
                best, bestE = mel.copy(), E
        T *= alpha
        if it % 500 == 0:
            trace.append((it, E, bestE))
    return best, bestE, trace


def synthesize(mel, path, sa_hz=138.59, beat=0.42, sr=22050):
    """Simple additive synthesis with a Sa-Pa tanpura-like drone."""
    out = []
    for i, n in enumerate(mel):
        f = sa_hz * 2 ** (SEMI[n] / 12)
        t = np.arange(int(sr * beat)) / sr
        env = np.minimum(1, t / 0.02) * np.exp(-2.2 * t)
        tone = sum(a * np.sin(2 * np.pi * f * h * t) for h, a in [(1, 1), (2, .45), (3, .25), (4, .12)])
        # gentle oscillation (andolan) on komal r and d
        if NAMES[n] in ("r", "d", "d."):
            tone = np.sin(2 * np.pi * f * t + 0.6 * np.sin(2 * np.pi * 3.0 * t) * f / 60)
        out.append(env * tone)
    sig = np.concatenate(out)
    t = np.arange(len(sig)) / sr
    drone = 0.12 * (np.sin(2 * np.pi * sa_hz / 2 * t) + 0.6 * np.sin(2 * np.pi * sa_hz * 0.75 * t))
    sig = sig / np.abs(sig).max() * 0.6 + drone
    wavfile.write(path, sr, (sig / np.abs(sig).max() * 32000).astype(np.int16))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    results = []
    for seed in range(8):
        mel, E, trace = anneal(np.random.default_rng(seed))
        results.append((E, mel, trace))
    results.sort(key=lambda r: r[0])
    E, mel, trace = results[0]
    total, parts, found = energy(mel, detail=True)
    phrases = [" ".join(NAMES[i] for i in mel[k:k + PHRASE_LEN]) for k in range(0, LENGTH, PHRASE_LEN)]
    print(f"best energy {total:.2f}  components {parts}")
    print("characteristic phrases present:", found)
    for p in phrases:
        print("  | " + p + " |")
    synthesize(mel, os.path.join(RESULTS, "bhairav_sa.wav"))
    json.dump(dict(energy=total, components={k: float(v) for k, v in parts.items()},
                   phrases_found=found, melody=phrases,
                   restart_energies=[float(r[0]) for r in results]),
              open(os.path.join(RESULTS, "lab4_bhairav_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
