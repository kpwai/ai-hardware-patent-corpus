#!/usr/bin/env python3
"""
STAGE 3 -- Multi-view fusion and clustering.

Three views, each L2-normalised, then explicitly weighted. Because every block
is unit-norm, the weights ARE the relative contribution to the cosine numerator
-- unlike v1, where raw embeddings of norm 6.84 sat beside a unit-norm CPC block
and a [0,1] year scalar, making "fusion" about 2% CPC and 2% time.

Clustering runs in a 10-D UMAP space with min_dist=0. The 2-D map is produced
separately, for the figure only. Density structure in 2-D is badly distorted by
crowding, and min_dist>0 deliberately spreads points apart.

    python stage3_cluster.py              # single configuration
    python stage3_cluster.py --ablate     # the weight sweep for the paper

Outputs:
    stage3_clusters/clustered.parquet
    stage3_clusters/ablation.csv
    stage3_clusters/manifest.json
"""

import argparse
import json
import platform
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from sklearn.preprocessing import MultiLabelBinarizer

import config as C


def l2n(X, eps=1e-9):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


def load_aligned():
    """Join corpus to embeddings on Lens ID. No positional assumptions."""
    corpus = pd.read_parquet(C.STAGE1 / "corpus.parquet")
    emb = np.load(C.STAGE2 / "embeddings.npy")
    ids = pd.read_csv(C.STAGE2 / "embedding_ids.csv")[C.ID_COL].astype(str)

    if len(ids) != emb.shape[0]:
        raise SystemExit(f"FATAL: {len(ids)} ids vs {emb.shape[0]} embedding rows")

    pos = pd.Series(np.arange(len(ids)), index=ids.values)
    key = corpus[C.ID_COL].astype(str)
    hit = key.isin(pos.index)
    if not hit.all():
        print(f"  WARNING: {(~hit).sum():,} corpus records lack an embedding; dropped")
        corpus = corpus.loc[hit].copy()
        key = key.loc[hit]
    emb = emb[pos.loc[key.values].values]

    assert len(corpus) == emb.shape[0]
    print(f"  Aligned on {C.ID_COL}: {len(corpus):,} records, dim {emb.shape[1]}")
    return corpus.reset_index(drop=True), emb


def build_views(corpus, emb):
    V_sem = l2n(np.asarray(emb, dtype=np.float32))

    g = C.CPC_GRANULARITY
    codes = corpus["cpc_codes"].fillna("").astype(str).map(
        lambda s: sorted({c.strip()[:g] for c in s.split(";;")
                          if c.strip() and len(c.strip()) >= g}))
    mlb = MultiLabelBinarizer()
    V_cpc = l2n(mlb.fit_transform(codes).astype(np.float32))

    yr = pd.to_numeric(corpus[C.YEAR_COL_PRIMARY], errors="coerce")
    n_missing = int(yr.isna().sum())
    if n_missing:
        print(f"  {n_missing:,} records lack {C.YEAR_COL_PRIMARY}; "
              f"imputed with the median (NOT a sentinel year)")
        yr = yr.fillna(yr.median())
    lo, hi = yr.min(), yr.max()
    V_time = ((yr - lo) / max(hi - lo, 1)).values.reshape(-1, 1).astype(np.float32)

    print(f"  Views: semantic {V_sem.shape} | cpc {V_cpc.shape} "
          f"({len(mlb.classes_)} codes) | time {V_time.shape}")
    print(f"  Year span: {int(lo)}-{int(hi)} on {C.YEAR_COL_PRIMARY}")
    print(f"  Mean CPC codes per patent: {(V_cpc > 0).sum(axis=1).mean():.2f}")
    return V_sem, V_cpc, V_time


def assemble(V_sem, V_cpc, V_time, w):
    blocks = []
    for name, V in (("semantic", V_sem), ("cpc", V_cpc), ("temporal", V_time)):
        if w.get(name, 0) > 0:
            blocks.append((V * w[name]).astype(np.float32))
    X = np.hstack(blocks)
    tot = sum(v for v in w.values() if v > 0)
    desc = ", ".join(f"{k}={v:.2f} ({100*v/tot:.0f}%)"
                     for k, v in w.items() if v > 0)
    print(f"    weights: {desc} | matrix {X.shape} ({X.nbytes/1e9:.2f} GB)")
    return X


def run_cluster(X, seed):
    import umap
    from sklearn.cluster import HDBSCAN

    t0 = time.time()
    Z = umap.UMAP(random_state=seed, **C.UMAP_PARAMS).fit_transform(X)
    t_u = time.time() - t0

    t0 = time.time()
    labels = HDBSCAN(n_jobs=-1, **C.HDBSCAN_PARAMS).fit_predict(Z)
    t_h = time.time() - t0

    m = labels != -1
    k = int(len(set(labels[m])))
    noise = float((~m).mean())
    sil = np.nan
    if k > 1 and m.sum() > 1000:
        idx = np.random.default_rng(seed).choice(np.flatnonzero(m),
                                                 min(20000, int(m.sum())),
                                                 replace=False)
        sil = float(silhouette_score(Z[idx], labels[idx]))
    print(f"    UMAP {t_u/60:.1f}m | HDBSCAN {t_h/60:.1f}m | "
          f"{k} clusters | noise {100*noise:.1f}% | silhouette {sil:.3f}")
    return Z, labels, {"umap_min": t_u/60, "hdbscan_min": t_h/60,
                       "n_clusters": k, "noise_frac": noise, "silhouette": sil}


def eta_squared(years, labels):
    """Share of year variance explained by cluster membership. High values mean
    the partition is tracking time -- the circularity the reviewer flagged."""
    m = labels != -1
    if m.sum() < 10:
        return np.nan
    g = pd.DataFrame({"y": np.asarray(years)[m], "c": labels[m]}).dropna()
    if g.empty:
        return np.nan
    grand = g["y"].mean()
    ssb = g.groupby("c")["y"].apply(lambda s: len(s) * (s.mean() - grand) ** 2).sum()
    sst = ((g["y"] - grand) ** 2).sum()
    return float(ssb / sst) if sst > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablate", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print("STAGE 3 -- MULTI-VIEW CLUSTERING")
    print("=" * 72)
    corpus, emb = load_aligned()
    V_sem, V_cpc, V_time = build_views(corpus, emb)
    years = pd.to_numeric(corpus[C.YEAR_COL_PRIMARY], errors="coerce").values

    env = {
        "stage": 3, "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "n_records": len(corpus), "embedding_dim": int(V_sem.shape[1]),
        "cpc_dim": int(V_cpc.shape[1]), "cpc_granularity": C.CPC_GRANULARITY,
        "year_col": C.YEAR_COL_PRIMARY, "seed": C.SEED,
        "umap": C.UMAP_PARAMS, "hdbscan": C.HDBSCAN_PARAMS,
    }

    if args.ablate:
        print("\n" + "-" * 72)
        print("ABLATION -- does the temporal view manufacture the clusters?")
        print("-" * 72)
        results, label_sets = [], {}

        for wt in C.ABLATION_TIME_WEIGHTS:
            w = dict(C.VIEW_WEIGHTS); w["temporal"] = wt
            print(f"  temporal={wt:.2f}")
            X = assemble(V_sem, V_cpc, V_time, w)
            _, lab, meta = run_cluster(X, C.SEED)
            label_sets[("time", wt)] = lab
            results.append({"view": "temporal", "weight": wt,
                            "eta2_year": eta_squared(years, lab), **meta})
            del X

        for wc in C.ABLATION_CPC_WEIGHTS:
            w = dict(C.VIEW_WEIGHTS); w["cpc"] = wc
            print(f"  cpc={wc:.2f}")
            X = assemble(V_sem, V_cpc, V_time, w)
            _, lab, meta = run_cluster(X, C.SEED)
            label_sets[("cpc", wc)] = lab
            results.append({"view": "cpc", "weight": wc,
                            "eta2_year": eta_squared(years, lab), **meta})
            del X

        base_t = label_sets[("time", 0.0)]
        base_c = label_sets[("cpc", 0.0)]
        for r in results:
            base = base_t if r["view"] == "temporal" else base_c
            lab = label_sets[(("time" if r["view"] == "temporal" else "cpc"),
                              r["weight"])]
            r["ARI_vs_zero"] = adjusted_rand_score(base, lab)
            r["NMI_vs_zero"] = normalized_mutual_info_score(base, lab)

        out = pd.DataFrame(results)
        out.to_csv(C.STAGE3 / "ablation.csv", index=False)
        print("\n" + out[["view", "weight", "n_clusters", "noise_frac",
                          "silhouette", "eta2_year", "ARI_vs_zero"]]
              .round(3).to_string(index=False))
        print("\n  Reading the table:")
        print("   * ARI_vs_zero near 1.0 as weight rises -> that view barely")
        print("     changes the partition; the fusion claim is weak for it.")
        print("   * ARI falling AND eta2_year climbing with temporal weight ->")
        print("     time is manufacturing clusters; growth findings are circular.")
        print("   * Report the configuration where substantive findings survive")
        print("     at weight 0. That is the defensible one.")
        env["ablation"] = results
        (C.STAGE3 / "manifest.json").write_text(json.dumps(env, indent=2, default=str))
        return

    # ---- single configuration -------------------------------------------
    print("\n" + "-" * 72)
    print("CLUSTERING")
    print("-" * 72)
    X = assemble(V_sem, V_cpc, V_time, C.VIEW_WEIGHTS)
    Z, labels, meta = run_cluster(X, C.SEED)
    corpus["tech_cluster"] = labels

    import umap
    print("  2-D projection for the figure only...")
    Z2 = umap.UMAP(random_state=C.SEED, **C.UMAP_VIZ_PARAMS).fit_transform(X)
    corpus["umap_1"], corpus["umap_2"] = Z2[:, 0], Z2[:, 1]

    np.save(C.STAGE3 / "cluster_space.npy", Z)
    np.save(C.STAGE3 / "viz_2d.npy", Z2)
    corpus.to_parquet(C.STAGE3 / "clustered.parquet", index=False)

    env["weights"] = C.VIEW_WEIGHTS
    env["result"] = meta
    env["eta2_year"] = eta_squared(years, labels)
    (C.STAGE3 / "manifest.json").write_text(json.dumps(env, indent=2, default=str))

    print(f"\n  eta^2 (year variance explained by cluster): {env['eta2_year']:.3f}")
    print(f"  Wrote -> {C.STAGE3}")
    print("=" * 72)


if __name__ == "__main__":
    main()
