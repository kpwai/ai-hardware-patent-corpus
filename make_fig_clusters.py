#!/usr/bin/env python3
"""
Figure 3 -- 2-D projection of the corpus, coloured by cluster, with the largest
clusters annotated by their top TF-IDF terms.

Also writes cluster_names.csv, a table of every cluster with size, median filing
year and top terms -- useful for the paper whether or not the figure is used.

    python make_fig_clusters.py
    python make_fig_clusters.py --annotate 8

Full-column width (7.0 in) is the default since the annotations need room; pass
--single for a 3.4 in version.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import config as C

FIGDIR = C.ROOT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# Manual names take precedence over TF-IDF terms where you have chosen one.
# Fill these in after reading cluster_names.csv; leave empty to use top terms.
MANUAL = {
    30: "Neuromorphic devices",
    15: "Neural accelerators",
    32: "Memristive circuits",
    11: "Processing-in-memory",
    23: "Neural processing units",
    25: "Accelerator dataflow",
}


def top_terms(texts, k=4):
    v = TfidfVectorizer(max_features=4000, stop_words="english",
                        ngram_range=(1, 2), min_df=3)
    try:
        X = v.fit_transform(texts)
    except ValueError:
        return []
    terms = np.array(v.get_feature_names_out())
    scores = np.asarray(X.mean(axis=0)).ravel()
    return list(terms[scores.argsort()[::-1][:k]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotate", type=int, default=6,
                    help="how many of the largest clusters to label")
    ap.add_argument("--single", action="store_true",
                    help="single-column width instead of full width")
    args = ap.parse_args()

    d = pd.read_parquet(C.STAGE3 / "clustered.parquet")
    if "umap_1" not in d.columns:
        raise SystemExit("clustered.parquet has no umap_1/umap_2; "
                         "re-run stage3_cluster.py")

    clustered = d[d.tech_cluster != -1]
    noise = d[d.tech_cluster == -1]
    sizes = clustered.tech_cluster.value_counts()

    # ---- cluster table (useful independently of the figure) --------------
    rows = []
    for c, g in clustered.groupby("tech_cluster"):
        rows.append({
            "cluster": int(c),
            "n": len(g),
            "pct": 100 * len(g) / len(d),
            "median_filing_year": float(pd.to_numeric(
                g[C.YEAR_COL_PRIMARY], errors="coerce").median()),
            "top_terms": ", ".join(top_terms(g.text.fillna(""))),
        })
    tab = pd.DataFrame(rows).sort_values("n", ascending=False)
    out = C.STAGE3 / "cluster_names.csv"
    tab.to_csv(out, index=False)
    print(f"  Wrote {out} ({len(tab)} clusters)")
    print(tab.head(12).to_string(index=False))

    # ---- figure ----------------------------------------------------------
    w = 3.4 if args.single else 7.0
    fig, ax = plt.subplots(figsize=(w, w * 0.62))

    ax.scatter(noise.umap_1, noise.umap_2, s=1.0, c="0.88", alpha=0.35,
               lw=0, rasterized=True, zorder=1)

    # tab20 alternates light/dark pairs, so half the clusters wash out at
    # small point sizes. Interleave two saturated qualitative maps instead.
    base = ([plt.get_cmap("tab10")(i) for i in range(10)] +
            [plt.get_cmap("Dark2")(i) for i in range(8)] +
            [plt.get_cmap("Set1")(i) for i in range(9)])
    for i, c in enumerate(sorted(clustered.tech_cluster.unique())):
        g = clustered[clustered.tech_cluster == c]
        ax.scatter(g.umap_1, g.umap_2, s=5.0, color=base[i % len(base)],
                   alpha=1.0, lw=0, rasterized=True, zorder=2)

    for c in sizes.head(args.annotate).index:
        g = clustered[clustered.tech_cluster == c]
        x, y = g.umap_1.median(), g.umap_2.median()
        label = MANUAL.get(int(c))
        if not label:
            terms = top_terms(g.text.fillna(""), k=2)
            label = ", ".join(terms) if terms else f"C{c}"
        ax.annotate(label, (x, y), fontsize=7, ha="center", va="center",
                    zorder=4, color="0.1", fontweight="medium",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec="0.35", lw=0.6, alpha=1.0))

    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel(
        f"{len(sizes)} clusters, {100*len(noise)/len(d):.0f}% unclustered "
        f"(grey)", fontsize=7)

    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"fig_clusters.{ext}", dpi=300,
                    bbox_inches="tight")
    print(f"\n  wrote {FIGDIR}/fig_clusters.{{pdf,png}}")
    print("  Review cluster_names.csv, then fill MANUAL in this script with")
    print("  readable names for the annotated clusters and re-run.")


if __name__ == "__main__":
    main()