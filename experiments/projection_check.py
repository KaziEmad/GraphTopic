"""
Projection check for Reviewer 3, Concern 6.

Question: does projecting two latent subtopics onto the line through their
centres (the 1-D overlap omega used for the merge decision) preserve the overlap
information that exists before the projection?

How it works
------------
GraphTopic is run exactly as in full_dataset_comparison_v1.py (same code, same
embedding model, same chunks, same selected hyperparameters). Every call to
get_overlap() is intercepted: the original 1-D omega is kept unchanged and, for
the same pair of subtopics, three reference measures are computed from the
embeddings *before* the projection:

  ovl_3d     overlap coefficient  integral(min(p, q))  of the two Gaussians fitted
             in the same 3-D PCA space (Monte-Carlo; the direct 3-D analogue
             of omega)
  knn_mix    k-nearest-neighbour mixing in the full 768-D embedding space
             (1 = the two subtopics are fully interleaved, 0 = fully separated)
  cos_cent   cosine similarity of the two 768-D subtopic centroids

The script reports Spearman/Pearson correlations between omega and each
reference, and how often the merge decision (omega >= theta) agrees with the
decision taken on ovl_3d (ovl_3d >= theta). It also draws a scatter plot and
one worked example (a merged pair and a non-merged pair).

Usage
-----
Put this file next to full_dataset_comparison_v1.py (the Test_8 folder) and run

    python projection_check.py
    python projection_check.py --datasets SetFit/bbc-news --max-pairs 2000

Results are written to projection_check_results/.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import full_dataset_comparison_v1 as fdc  # noqa: E402  (runs its env setup)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402

OUT_DIR = HERE / "projection_check_results"

# First chunk of each dataset, with the (N, eta, theta) selected for that chunk
# in full_comparison_v1 (global_comparison_summary.csv). TopM = N, TopN = eta.
DEFAULT_RUNS = {
    "SetFit/bbc-news": dict(TopM=1500, TopN=15, overlapping_threshold=0.35),
    "SetFit/20_newsgroups": dict(TopM=1500, TopN=10, overlapping_threshold=0.75),
    "TrumpTweet": dict(TopM=1500, TopN=10, overlapping_threshold=0.35),
}

SEED = 42
MC_SAMPLES = 20000
RIDGE = 1e-6

# ---------------------------------------------------------------------------
# Reference overlap measures
# ---------------------------------------------------------------------------


def _gauss(X):
    mu = X.mean(axis=0)
    if X.shape[0] > 1:
        cov = np.cov(X, rowvar=False)
    else:
        cov = np.zeros((X.shape[1], X.shape[1]))
    cov = np.atleast_2d(cov) + RIDGE * np.eye(X.shape[1])
    return mu, cov


def _logpdf(X, mu, cov):
    d = X.shape[1]
    sign, logdet = np.linalg.slogdet(cov)
    inv = np.linalg.inv(cov)
    diff = X - mu
    m = np.einsum("ij,jk,ik->i", diff, inv, diff)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + m)


def overlap_coefficient_mc(C1, C2, rng):
    """integral(min(p, q)) = E_p[min(1, q/p)] for Gaussians fitted to C1, C2."""
    mu1, cov1 = _gauss(C1)
    mu2, cov2 = _gauss(C2)
    X = rng.multivariate_normal(mu1, cov1, size=MC_SAMPLES)
    ratio = np.exp(np.clip(_logpdf(X, mu2, cov2) - _logpdf(X, mu1, cov1), -50, 50))
    return float(np.mean(np.minimum(1.0, ratio)))


def knn_mixing(E, labels, k=5):
    """Fraction of k-NN (cosine) from the other subtopic, normalised so that a
    random interleaving gives 1 and complete separation gives 0."""
    n = E.shape[0]
    if n < 3:
        return np.nan
    k = min(k, n - 1)
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    S = En @ En.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argsort(-S, axis=1)[:, :k]
    other = (labels[nn] != labels[:, None]).mean(axis=1)
    n1 = (labels == labels[0]).sum()
    expected = np.where(labels == labels[0], (n - n1) / (n - 1), n1 / (n - 1))
    return float(np.clip(np.mean(other / expected), 0.0, 1.0))


def cosine_centroids(C1, C2):
    a, b = C1.mean(axis=0), C2.mean(axis=0)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ---------------------------------------------------------------------------
# Interception of GraphTopic's get_overlap
# ---------------------------------------------------------------------------

RECORDS = []
EXAMPLES = {}  # pair data kept for the worked-example figure
STATE = {"dataset": None, "theta": None, "max_pairs": None, "seen": 0}
_original_get_overlap = fdc.get_overlap
_rng = np.random.default_rng(SEED)


def recording_get_overlap(indices, X_df):
    omega = _original_get_overlap(indices, X_df)  # unchanged merge decision
    STATE["seen"] += 1
    if STATE["max_pairs"] and len(
        [r for r in RECORDS if r["dataset"] == STATE["dataset"]]
    ) >= STATE["max_pairs"]:
        return omega
    try:
        idx1, idx2 = indices
        sub = X_df[X_df["topic"].isin(indices)]
        E = sub.drop(columns=["topic"]).to_numpy(dtype=float)
        labels = sub["topic"].to_numpy()
        # same 3-D reduction as get_overlap (fdc.reducer = PCA(3, random_state=45))
        R = PCA(n_components=3, random_state=45).fit_transform(E)
        C1, C2 = R[labels == idx1], R[labels == idx2]
        rec = {
            "dataset": STATE["dataset"],
            "theta": STATE["theta"],
            "n1": int(C1.shape[0]),
            "n2": int(C2.shape[0]),
            "omega_1d": float(omega),
            "ovl_3d": overlap_coefficient_mc(C1, C2, _rng),
            "knn_mix_768d": knn_mixing(E, labels),
            "cos_centroid_768d": cosine_centroids(E[labels == idx1], E[labels == idx2]),
        }
        RECORDS.append(rec)
        key = "merge" if omega >= STATE["theta"] else "keep"
        # keep the clearest example of each kind with at least 5 keyphrases per side
        if min(C1.shape[0], C2.shape[0]) >= 5:
            best = EXAMPLES.get(key)
            score = omega if key == "merge" else -omega
            if best is None or score > best["score"]:
                EXAMPLES[key] = dict(score=score, R=R, labels=labels, idx=(idx1, idx2), rec=rec)
    except Exception as exc:  # never break the GraphTopic run
        print(f"  [projection_check] skipped a pair: {exc}")
    return omega


fdc.get_overlap = recording_get_overlap

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def first_chunk(dataset_name):
    cfg_entry = next(d for d in fdc.config.DATASETS if d["name"] == dataset_name)
    docs = fdc.clean_docs(fdc.load_dataset_docs(cfg_entry))
    chunks, _ = fdc.make_fixed_doc_chunks(docs, dataset_name, n_chunks=fdc.N_EQUAL_CHUNKS)
    return chunks[0]


def summarise(df):
    out = {}
    for ds, g in list(df.groupby("dataset")) + [("ALL", df)]:
        g = g.dropna()
        if len(g) < 5:
            continue
        th = g["theta"].to_numpy()
        dec_1d = (g["omega_1d"] >= th).astype(int)
        dec_3d = (g["ovl_3d"] >= th).astype(int)
        res = {"pairs": int(len(g))}
        for ref in ["ovl_3d", "knn_mix_768d", "cos_centroid_768d"]:
            rho, p_rho = spearmanr(g["omega_1d"], g[ref])
            r, p_r = pearsonr(g["omega_1d"], g[ref])
            res[ref] = dict(spearman=float(rho), spearman_p=float(p_rho),
                            pearson=float(r), pearson_p=float(p_r))
        res["decision_agreement_vs_ovl_3d"] = float((dec_1d == dec_3d).mean())
        res["cohen_kappa_vs_ovl_3d"] = (
            float(cohen_kappa_score(dec_1d, dec_3d)) if dec_1d.nunique() > 1 or dec_3d.nunique() > 1 else None
        )
        res["merged_pairs_1d"] = int(dec_1d.sum())
        out[ds] = res
    return out


def plot_scatter(df, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    refs = [("ovl_3d", "Overlap in 3-D PCA space (before projection)"),
            ("knn_mix_768d", "k-NN mixing in 768-D embedding space"),
            ("cos_centroid_768d", "Centroid cosine similarity (768-D)")]
    for ax, (ref, label) in zip(axes, refs):
        for ds, g in df.groupby("dataset"):
            ax.scatter(g[ref], g["omega_1d"], s=8, alpha=0.45, label=ds.split("/")[-1])
        g = df.dropna(subset=[ref])
        rho = spearmanr(g["omega_1d"], g[ref])[0]
        ax.set_xlabel(label)
        ax.set_ylabel(r"1-D overlap $\omega$ (used by GraphTopic)")
        ax.set_title(f"Spearman $\\rho$ = {rho:.2f}")
        if ref == "ovl_3d":
            ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].legend(fontsize=8, markerscale=2)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _norm_pdf(x, m, v):
    v = max(v, 1e-12)
    return np.exp(-((x - m) ** 2) / (2 * v)) / np.sqrt(2 * np.pi * v)


def plot_example(path):
    if not EXAMPLES:
        return
    keys = [k for k in ("merge", "keep") if k in EXAMPLES]
    fig, axes = plt.subplots(2, len(keys), figsize=(6 * len(keys), 8.5), squeeze=False)
    for col, key in enumerate(keys):
        ex = EXAMPLES[key]
        R, labels, (i1, i2), rec = ex["R"], ex["labels"], ex["idx"], ex["rec"]
        C1, C2 = R[labels == i1], R[labels == i2]
        c1, c2 = C1.mean(axis=0), C2.mean(axis=0)
        u = (c2 - c1) / np.linalg.norm(c2 - c1)
        p1, p2 = (C1 - c1) @ u, (C2 - c1) @ u
        ax = axes[0, col]
        ax.scatter(C1[:, 0], C1[:, 1], s=18, label="subtopic A")
        ax.scatter(C2[:, 0], C2[:, 1], s=18, label="subtopic B")
        ax.plot([c1[0], c2[0]], [c1[1], c2[1]], "k-", lw=1.5, label="centre line")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_title(f"{'Merged' if key == 'merge' else 'Not merged'} pair "
                     f"(3-D overlap = {rec['ovl_3d']:.2f})")
        ax.legend(fontsize=8)
        ax = axes[1, col]
        xs = np.linspace(min(p1.min(), p2.min()) - 0.3, max(p1.max(), p2.max()) + 0.3, 400)
        f1, f2 = _norm_pdf(xs, p1.mean(), p1.var()), _norm_pdf(xs, p2.mean(), p2.var())
        ax.plot(xs, f1, label="A projected")
        ax.plot(xs, f2, label="B projected")
        ax.fill_between(xs, np.minimum(f1, f2), color="grey", alpha=0.4, label=r"overlap $\omega$")
        ax.set_xlabel("position on the centre line")
        ax.set_title(rf"1-D overlap $\omega$ = {rec['omega_1d']:.2f} "
                     rf"($\theta$ = {rec['theta']})")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULT_RUNS))
    ap.add_argument("--max-pairs", type=int, default=3000,
                    help="max recorded pairs per dataset (all pairs are still evaluated)")
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(exist_ok=True)

    device = fdc.detect_device()
    emb_model = fdc._load_embedding_model(fdc.config.FIXED_PARAMS["embedding_model"], device)
    fdc.embeddingModel = emb_model

    for ds in args.datasets:
        params = DEFAULT_RUNS.get(ds, dict(TopM=1500, TopN=15, overlapping_threshold=0.35))
        STATE.update(dataset=ds, theta=params["overlapping_threshold"], max_pairs=args.max_pairs)
        print(f"\n=== {ds}  params={params}")
        t0 = time.time()
        chunk = first_chunk(ds)
        scores = fdc.run_graphtopic_on_chunk(chunk, params, emb_model, fdc.config)
        n = len([r for r in RECORDS if r["dataset"] == ds])
        print(f"  status={scores.get('status')}  topics={scores.get('num_topics')}  "
              f"pairs recorded={n}  ({time.time() - t0:.0f}s)")
        pd.DataFrame(RECORDS).to_csv(OUT_DIR / "pairs.csv", index=False)  # checkpoint

    df = pd.DataFrame(RECORDS)
    if df.empty:
        print("No pairs were recorded (no merging step was reached).")
        return
    summary = summarise(df)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    plot_scatter(df, OUT_DIR / "omega_vs_full_space")
    plot_example(OUT_DIR / "projection_example")

    print("\n=== Summary")
    print(json.dumps(summary, indent=2))
    print(f"\nWritten to {OUT_DIR}: pairs.csv, summary.json, "
          "omega_vs_full_space.png/pdf, projection_example.png/pdf")


if __name__ == "__main__":
    main()
