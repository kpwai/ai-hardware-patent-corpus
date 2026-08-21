#!/usr/bin/env python3
"""
HDBSCAN parameter sweep -- run against the CURRENT config.VIEW_WEIGHTS.

The embedding geometry depends on the view weights, so a sweep is only valid
for the weights it was run under. Sweeping with temporal=0.15 and then setting
temporal=0 produced a degenerate 2-cluster solution, because min_samples=5 is
permissive enough that without the temporal component the whole space reads as
one dense region.

This builds the UMAP embedding once from the weights currently in config.py,
then sweeps HDBSCAN over it.

    python sweep_hdbscan.py
"""

import warnings

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

import config as C
import stage3_cluster as S

MCS = (75, 100, 150, 200, 300)
MS = (5, 10, 20)


def main():
    print("=" * 66)
    print("HDBSCAN SWEEP -- weights:", C.VIEW_WEIGHTS)
    print("=" * 66)

    corpus, emb = S.load_aligned()
    Vs, Vc, Vt = S.build_views(corpus, emb)
    X = S.assemble(Vs, Vc, Vt, C.VIEW_WEIGHTS)

    import umap
    print("\n  Building UMAP embedding once...")
    Z = umap.UMAP(random_state=C.SEED, **C.UMAP_PARAMS).fit_transform(X)

    years = np.asarray(corpus[C.YEAR_COL_PRIMARY].astype(float))

    print(f"\n  {'mcs':>5} {'ms':>4} {'k':>5} {'noise%':>8} {'sil':>7} "
          f"{'largest%':>9} {'eta2':>7}")
    print("  " + "-" * 52)
    rows = []
    for mcs in MCS:
        for ms in MS:
            lab = HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                          n_jobs=-1).fit_predict(Z)
            m = lab != -1
            k = len(set(lab[m]))
            if k < 2:
                print(f"  {mcs:>5} {ms:>4} {k:>5} {'':>8} "
                      f"{'DEGENERATE':>17}")
                continue
            sil = silhouette_score(Z[m], lab[m])
            sizes = np.bincount(lab[m])
            largest = 100 * sizes.max() / len(lab)
            eta2 = S.eta_squared(years, lab)
            print(f"  {mcs:>5} {ms:>4} {k:>5} {100*(~m).mean():>7.1f} "
                  f"{sil:>7.3f} {largest:>8.1f} {eta2:>7.3f}")
            rows.append(dict(mcs=mcs, ms=ms, k=k, noise=100*(~m).mean(),
                             sil=sil, largest_pct=largest, eta2=eta2))

    import pandas as pd
    out = C.STAGE3 / "hdbscan_sweep.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n  Wrote {out}")
    print("\n  CHOOSING:")
    print("   * reject any row where one cluster holds >40% of the corpus --")
    print("     that is a giant blob with satellites, not a partition")
    print("   * k in roughly 15-40 keeps clusters interpretable and namable")
    print("   * noise has a floor; do not chase it below what the data allows")
    print("   * prefer higher silhouette among rows that pass the above")


if __name__ == "__main__":
    main()
