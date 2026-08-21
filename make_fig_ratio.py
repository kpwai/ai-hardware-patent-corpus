#!/usr/bin/env python3
"""
Figure 1 -- measurement bias: naive subclass sweep vs validated corpus.

Panel (a) shows measured prevalence under both constructions on a log scale.
Panel (b) shows the understatement factor under the filing-year basis only.
Publication-year values are NOT plotted: they give the same rank ordering
(Spearman rho = 1.0) but larger magnitudes, and plotting both made the bar
labels ambiguous. State the replication in the caption instead.

    python make_fig_ratio.py [filing_csv] [pub_csv]
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGDIR = Path("output/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

fil_path = sys.argv[1] if len(sys.argv) > 1 else \
    "output/stage4_analysis/naive_vs_validated_summary_filing_year.csv"
pub_path = sys.argv[2] if len(sys.argv) > 2 else \
    "output/stage4_analysis/naive_vs_validated_summary_pub_year.csv"

fil = pd.read_csv(fil_path)
d = fil.sort_values("ratio")

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4),
                         gridspec_kw={"width_ratios": [1.25, 1]})

# ---- (a) prevalence under both constructions, log scale -----------------
ax = axes[0]
y = np.arange(len(d))
ax.hlines(y, d.naive_pooled_pct, d.validated_pooled_pct,
          color="0.75", lw=2, zorder=1)
ax.scatter(d.naive_pooled_pct, y, s=40, color="#b03a2e",
           label="Naive subclass sweep", zorder=2)
ax.scatter(d.validated_pooled_pct, y, s=40, color="#1a5276",
           label="Validated corpus", zorder=2)
ax.set_yticks(y)
ax.set_yticklabels(d.architecture, fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("Share of corpus mentioning architecture (%, log scale)",
              fontsize=8)
ax.tick_params(axis="x", labelsize=7)
ax.grid(axis="x", alpha=0.3, ls=":", lw=0.5)
ax.set_title("(a) Measured share by corpus construction", fontsize=8,
             loc="left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# ---- (b) understatement factor, filing year only ------------------------
ax = axes[1]
ax.barh(y, d.ratio, height=0.62, color="#1a5276")
for yy, v in zip(y, d.ratio):
    ax.text(v + 1.5, yy, f"{v:.0f}x", va="center", fontsize=7.5)
ax.set_yticks(y)
ax.set_yticklabels([])
ax.set_xlabel("Prevalence ratio (validated / sweep)", fontsize=8)
ax.set_xlim(0, d.ratio.max() * 1.16)
ax.tick_params(axis="x", labelsize=7)
ax.axvline(1, color="0.4", lw=0.8)
ax.grid(axis="x", alpha=0.3, ls=":", lw=0.5)
ax.set_title("(b) Bias scales with AI-specificity", fontsize=8, loc="left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.tight_layout(pad=0.4)
for ext in ("pdf", "png"):
    fig.savefig(FIGDIR / f"fig_naive_vs_validated.{ext}", dpi=300,
                bbox_inches="tight")
print(f"wrote {FIGDIR}/fig_naive_vs_validated.{{pdf,png}}")

# ---- robustness numbers for the caption ---------------------------------
try:
    pub = pd.read_csv(pub_path)
    m = fil.merge(pub, on="architecture", suffixes=("_fil", "_pub"))
    rho = m.ratio_fil.corr(m.ratio_pub, method="spearman")
    print(f"\n  Publication-year ratios: "
          f"{', '.join(f'{v:.1f}x' for v in m.sort_values('ratio_fil').ratio_pub)}")
    print(f"  Spearman rho vs filing year: {rho:.3f}")
    print("  Cite this in the caption rather than plotting a second series.")
except Exception as e:
    print(f"\n  (could not read {pub_path}: {e})")