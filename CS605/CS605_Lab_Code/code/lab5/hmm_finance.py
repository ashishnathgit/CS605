"""
CS605 Lab 5 - Market regimes with a Gaussian Hidden Markov Model.

Data:  daily adjusted close of the S&P 500 (^GSPC) and, for the bonus comparison,
       the NASDAQ Composite (^IXIC).  The script first tries the Yahoo Finance API
       (yfinance).  If Yahoo cannot be reached it falls back to the Yahoo-sourced
       daily series bundled with the `arch` package (1999-01-04 .. 2018-12-31),
       which is what produced the results in the report.
Model: observations x_t = 100 * log(P_t / P_{t-1}) (daily log return in %),
       hidden state z_t in {1..K}, x_t | z_t = k ~ N(mu_k, sigma_k^2),
       P(z_{t+1} = j | z_t = i) = A_ij,  fitted by Baum-Welch (EM) with hmmlearn,
       decoded with Viterbi.  K = 2, 3, 4 compared by log-likelihood, AIC and BIC.

Run:  python lab5/hmm_finance.py [--ticker ^GSPC --start 2005-01-01]
"""
import argparse
import json
import os
import warnings

import matplotlib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "results", "lab5")
PALETTE = {2: ["#2a9d8f", "#d1495b"], 3: ["#2a9d8f", "#e9a03b", "#d1495b"],
           4: ["#2a9d8f", "#8ab17d", "#e9a03b", "#d1495b"]}


# ----------------------------------------------------------------------------
# Part 1: data acquisition and preprocessing
# ----------------------------------------------------------------------------
def download(ticker, start, end):
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        if len(df) > 1000:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df[["Adj Close"]].rename(columns={"Adj Close": "adj_close"}), "Yahoo Finance (yfinance)"
    except Exception:
        pass
    # offline fallback: Yahoo-sourced data shipped with the `arch` package
    if ticker in ("^GSPC", "SPX"):
        from arch.data import sp500 as src
    elif ticker in ("^IXIC", "NASDAQ"):
        from arch.data import nasdaq as src
    else:
        raise RuntimeError("Yahoo Finance unreachable and no offline copy for " + ticker)
    df = src.load()[["Adj Close"]].rename(columns={"Adj Close": "adj_close"})
    return df.loc[start:end], "arch package (Yahoo Finance daily data, 1999-2018)"


def preprocess(df):
    df = df.copy()
    df = df[~df.index.duplicated()].sort_index()
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    n_missing = int(df["adj_close"].isna().sum())
    df["adj_close"] = df["adj_close"].ffill()
    df = df[df["adj_close"] > 0]
    df["ret"] = 100 * np.log(df["adj_close"]).diff()
    df = df.dropna()
    # flag (but keep) extreme days; drop only impossible values (data errors)
    bad = df["ret"].abs() > 30
    df = df[~bad]
    return df, dict(missing_filled=n_missing, dropped_errors=int(bad.sum()))


# ----------------------------------------------------------------------------
# Part 2: fitting the Gaussian HMM
# ----------------------------------------------------------------------------
def fit_hmm(x, k, n_init=10, seed=0):
    best, best_ll = None, -np.inf
    for i in range(n_init):
        m = GaussianHMM(n_components=k, covariance_type="diag", n_iter=500, tol=1e-5,
                        random_state=seed + i)
        try:
            m.fit(x)
            ll = m.score(x)
        except Exception:
            continue
        if ll > best_ll and np.all(np.isfinite(m.transmat_)):
            best, best_ll = m, ll
    return relabel_by_volatility(best), best_ll


def relabel_by_volatility(m):
    """Order the states by increasing variance so that state 0 = calmest regime."""
    order = np.argsort(m.covars_.reshape(m.n_components, -1)[:, 0])
    k = m.n_components
    new = GaussianHMM(n_components=k, covariance_type="diag")
    new.startprob_ = m.startprob_[order]
    new.transmat_ = m.transmat_[np.ix_(order, order)]
    new.means_ = m.means_[order]
    new.covars_ = m.covars_.reshape(k, -1)[order]
    new.n_features = 1
    return new


def n_params(k):
    return (k - 1) + k * (k - 1) + 2 * k


def regime_name(mu, sd, k):
    if k == 2:
        return ["low-vol / bull", "high-vol / bear"]
    names = {3: ["calm bull", "normal", "turbulent / crisis"],
             4: ["calm bull", "normal", "volatile", "crisis"]}
    return names.get(k, [f"state {i}" for i in range(k)])


def analyse(ticker, start, end, tag):
    raw, source = download(ticker, start, end)
    df, clean = preprocess(raw)
    x = df[["ret"]].values
    out = dict(ticker=ticker, source=source, start=str(df.index[0].date()), end=str(df.index[-1].date()),
               n_obs=len(df), cleaning=clean, models={})
    print(f"{ticker}: {len(df)} daily returns {out['start']} .. {out['end']} from {source}")
    fitted = {}
    for k in (2, 3, 4):
        m, ll = fit_hmm(x, k)
        aic = -2 * ll + 2 * n_params(k)
        bic = -2 * ll + n_params(k) * np.log(len(x))
        states = m.predict(x)
        post = m.predict_proba(x)
        mu = m.means_.ravel()
        sd = np.sqrt(m.covars_.ravel())
        A = m.transmat_
        # Part 5: forecast regime probabilities from the last filtered posterior
        p_last = post[-1]
        horizon = {h: (p_last @ np.linalg.matrix_power(A, h)).tolist() for h in (1, 5, 21, 63)}
        stat = np.linalg.eig(A.T)
        pi = np.real(stat[1][:, np.argmin(np.abs(stat[0] - 1))])
        pi = pi / pi.sum()
        occ = np.bincount(states, minlength=k) / len(states)
        periods = []
        seg_start = 0
        for t in range(1, len(states) + 1):
            if t == len(states) or states[t] != states[seg_start]:
                periods.append((str(df.index[seg_start].date()), str(df.index[t - 1].date()),
                                int(states[seg_start]), t - seg_start))
                seg_start = t
        long_periods = sorted([p for p in periods if p[2] == k - 1], key=lambda p: -p[3])[:6]
        fitted[k] = (m, states, post)
        out["models"][k] = dict(
            loglik=ll, aic=aic, bic=bic, means=mu.tolist(), stds=sd.tolist(),
            ann_mean=(mu * 252).tolist(), ann_vol=(sd * np.sqrt(252)).tolist(),
            transmat=A.tolist(), expected_duration_days=(1 / (1 - np.diag(A))).tolist(),
            stationary=pi.tolist(), occupancy=occ.tolist(), n_switches=int(np.sum(np.diff(states) != 0)),
            last_state=int(states[-1]), last_posterior=p_last.tolist(), forecast=horizon,
            names=regime_name(mu, sd, k), longest_high_vol_spells=long_periods)
        print(f"  K={k}: logL={ll:.1f} AIC={aic:.1f} BIC={bic:.1f} mu={np.round(mu,3)} sd={np.round(sd,3)}")
    return df, out, fitted


# ----------------------------------------------------------------------------
# Part 4: visualisation
# ----------------------------------------------------------------------------
def plot_regimes(df, states, k, names, path, title):
    COLORS = PALETTE[k]
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, gridspec_kw=dict(height_ratios=[1.3, 1]))
    for s in range(k):
        mask = states == s
        ax[0].scatter(df.index[mask], df["adj_close"].values[mask], s=1.2, color=COLORS[s], label=names[s])
        ax[1].scatter(df.index[mask], df["ret"].values[mask], s=1.2, color=COLORS[s])
    ax[0].set_yscale("log"); ax[0].set_ylabel("adjusted close (log)")
    ax[0].legend(markerscale=6, fontsize=8, loc="upper left"); ax[0].set_title(title, fontsize=10)
    ax[1].set_ylabel("daily log return (%)")
    for a in ax:
        a.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_posterior(df, post, k, names, path):
    COLORS = PALETTE[k]
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.stackplot(df.index, post.T, colors=COLORS[:k], labels=names, alpha=0.9)
    ax.set_ylim(0, 1); ax.set_ylabel("P(state | data)"); ax.legend(fontsize=7, loc="upper left", ncol=k)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_transmat(A, names, path):
    k = len(A)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    im = ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{A[i][j]:.3f}", ha="center", va="center",
                    color="white" if A[i][j] > 0.6 else "black", fontsize=8)
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels([f"to {i}" for i in range(k)], fontsize=8)
    ax.set_yticklabels([f"from {i}" for i in range(k)], fontsize=8)
    plt.colorbar(im, fraction=0.046); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_hist(df, m, path):
    COLORS = PALETTE[m.n_components]
    x = df["ret"].values
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(x, bins=200, density=True, color="#bbb", label="returns")
    g = np.linspace(-8, 8, 600)
    pi = np.bincount(m.predict(x[:, None]), minlength=m.n_components) / len(x)
    tot = 0
    for s in range(m.n_components):
        mu, sd = m.means_[s, 0], np.sqrt(m.covars_[s, 0])
        comp = pi[s] * np.exp(-(g - mu) ** 2 / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi))
        tot = tot + comp
        ax.plot(g, comp, color=COLORS[s], lw=1)
    ax.plot(g, tot, "k--", lw=1, label="HMM mixture")
    ax.set_xlim(-6, 6); ax.set_xlabel("daily log return (%)"); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="^GSPC")
    ap.add_argument("--start", default="1999-01-01")
    ap.add_argument("--end", default="2026-10-27")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    summary = {}
    for ticker, tag in ((args.ticker, "spx"), ("^IXIC", "ixic")):
        df, out, fitted = analyse(ticker, args.start, args.end, tag)
        summary[tag] = out
        for k in (2, 3):
            m, states, post = fitted[k]
            names = out["models"][k]["names"]
            plot_regimes(df, states, k, names, os.path.join(RESULTS, f"{tag}_regimes_k{k}.png"),
                         f"{ticker}: Viterbi-decoded regimes, K = {k}")
            plot_posterior(df, post, k, names, os.path.join(RESULTS, f"{tag}_posterior_k{k}.png"))
            plot_transmat(out["models"][k]["transmat"], names, os.path.join(RESULTS, f"{tag}_transmat_k{k}.png"))
        plot_hist(df, fitted[2][0], os.path.join(RESULTS, f"{tag}_hist_k2.png"))
        df.assign(state_k2=fitted[2][1], state_k3=fitted[3][1]).to_csv(
            os.path.join(RESULTS, f"{tag}_returns_states.csv"))
    with open(os.path.join(RESULTS, "lab5_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)


if __name__ == "__main__":
    main()
