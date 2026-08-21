#!/usr/bin/env python3
"""
Representation sweep -- when HDBSCAN is degenerate at every parameter, the
problem is the embedding geometry, not the clusterer.

At 7,464 records the corpus is homogeneous (all G06N3/06 neural hardware), so
UMAP with n_neighbors=30 and subclass-level CPC produces one dense blob. This
varies the things that actually shape the space:

  * CPC granularity   4 = subclass (G06N, 160 codes)
                      8 = main group (G06N3/063, many more, far sharper)
  * CPC weight        how much the structural view counts
  * UMAP n_neighbors  low = local structure preserved, high = global

For each representation it reports the best HDBSCAN result found.

    python rep_sweep.py
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings("ignore")

import config as C
import stage3_cluster as S

GRAN = (4, 8)
CPC_W = (0.30, 0.60, 1.00)
NEIGH = (10, 15, 30)
HDB = ((50, 5), (75, 5), (100, 10), (150, 10))


def l2n(X, eps=1e-9):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


def cpc_view(corpus, g):
    codes = corpus["cpc_codes"].fillna("").astype(str).map(
        lambda s: sorted({c.strip()[:g] for c in s.split(";;")
                          if c.strip() and len(c.strip()) >= g}))
    mlb = MultiLabelBinarizer()
    V = l2n(mlb.fit_transform(codes).astype(np.float32))
    return V, len(mlb.classes_)


def main():
    import umap

    corpus, emb = S.load_aligned()
    Vs = l2n(np.asarray(emb, dtype=np.float32))
    years = pd.to_numeric(corpus[C.YEAR_COL_PRIMARY], errors="coerce").values

    print("\n" + "=" * 78)
    print(f"{'gran':>5} {'codes':>6} {'w_cpc':>6} {'nn':>4} | "
          f"{'mcs/ms':>8} {'k':>4} {'noise%':>7} {'sil':>7} {'largest%':>9}")
    print("=" * 78)

    rows = []
    for g in GRAN:
        Vc, ncodes = cpc_view(corpus, g)
        for w in CPC_W:
            X = np.hstack([Vs, (Vc * w).astype(np.float32)])
            for nn in NEIGH:
                p = dict(C.UMAP_PARAMS)
                p["n_neighbors"] = nn
                Z = umap.UMAP(random_state=C.SEED, **p).fit_transform(X)
                best = None
                for mcs, ms in HDB:
                    lab = HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                                  n_jobs=-1).fit_predict(Z)
                    m = lab != -1
                    k = len(set(lab[m]))
                    if k < 3:
                        continue
                    sizes = np.bincount(lab[m])
                    largest = 100 * sizes.max() / len(lab)
                    if largest > 40:
                        continue
                    sil = silhouette_score(Z[m], lab[m])
                    rec = dict(gran=g, codes=ncodes, w_cpc=w, nn=nn,
                               mcs=mcs, ms=ms, k=k,
                               noise=100 * (~m).mean(), sil=sil,
                               largest=largest,
                               eta2=S.eta_squared(years, lab))
                    # prefer interpretable k, then silhouette
                    score = (15 <= k <= 40) * 100 + sil
                    if best is None or score > best[0]:
                        best = (score, rec)
                if best is None:
                    print(f"{g:>5} {ncodes:>6} {w:>6.2f} {nn:>4} | "
                          f"{'--':>8} {'degenerate at all HDBSCAN settings':>40}")
                    continue
                r = best[1]
                rows.append(r)
                print(f"{g:>5} {ncodes:>6} {w:>6.2f} {nn:>4} | "
                      f"{str(r['mcs'])+'/'+str(r['ms']):>8} {r['k']:>4} "
                      f"{r['noise']:>6.1f} {r['sil']:>7.3f} {r['largest']:>8.1f}")

    if not rows:
        print("\n  Nothing usable. The corpus may simply be too homogeneous to")
        print("  partition -- which is itself reportable: a validated, tightly")
        print("  scoped corpus has less internal structure than a noisy one.")
        return

    df = pd.DataFrame(rows).sort_values(
        ["k", "sil"], key=lambda s: s if s.name != "k"
        else (s.between(15, 40)).astype(int), ascending=False)
    out = C.STAGE3 / "rep_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\n  Wrote {out}")

    ok = df[(df.k.between(12, 45)) & (df.largest < 40)]
    if len(ok):
        b = ok.sort_values("sil", ascending=False).iloc[0]
        print("\n  RECOMMENDED:")
        print(f"    CPC_GRANULARITY = {int(b.gran)}")
        print(f"    VIEW_WEIGHTS cpc = {b.w_cpc}")
        print(f"    UMAP_PARAMS n_neighbors = {int(b.nn)}")
        print(f"    HDBSCAN_PARAMS = dict(min_cluster_size={int(b.mcs)}, "
              f"min_samples={int(b.ms)})")
        print(f"    -> {int(b.k)} clusters, {b.noise:.1f}% noise, "
              f"silhouette {b.sil:.3f}, largest {b.largest:.1f}%")
    else:
        print("\n  No configuration reached 12+ clusters. Report the coarse")
        print("  partition honestly rather than forcing structure that is")
        print("  not there.")


if __name__ == "__main__":
    main()
