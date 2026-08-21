#!/usr/bin/env python3
"""
Figure 2 -- architecture trajectories, counts and shares.

Two panels, deliberately. A share-only chart invites the reading that
neuromorphic research declined: its share falls from 81% to 10%. It did not.
Neuromorphic counts rise from 13 to 117 families over the same period; the
corpus grew 3.25x while neuromorphic grew 1.07x, so it lost share while growing.
The counts panel makes this visible; the shares panel shows composition change.

Years before 2016 are dropped from the shares panel: with 16 families filed in
2010 a single record moves a share by 6 percentage points, so the early swings
are sampling noise rather than history.

    python make_fig_trajectory.py            # full width, two panels
    python make_fig_trajectory.py --single   # single column, stacked
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C

FIGDIR = C.ROOT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

SHOW = ["TPU/NPU", "Neuromorphic/Memristive", "PIM/Memory-Centric",
        "GPU", "Analog/Photonic", "CPU"]
SHORT = {"TPU/NPU": "TPU/NPU", "Neuromorphic/Memristive": "Neuromorphic",
         "PIM/Memory-Centric": "PIM", "GPU": "GPU",
         "Analog/Photonic": "Analog/Photonic", "CPU": "CPU"}
COLOR = {"TPU/NPU": "#1a5276", "Neuromorphic/Memristive": "#b03a2e",
         "PIM/Memory-Centric": "#1e8449", "GPU": "#7d6608",
         "Analog/Photonic": "#6c3483", "CPU": "#5d6d7e"}

# Shares are only interpretable once annual counts are large enough that one
# record does not move the estimate by several points.
SHARE_START = 2016


def draw(ax, d, cats, value_fmt, start, ylab, band=False):
    for cat in cats:
        col = f"{cat}_{value_fmt}"
        if col not in d.columns:
            continue
        sub = d[d.year >= start]
        solid = sub[sub.year <= C.YEAR_END]
        ax.plot(solid.year, solid[col], marker="o", ms=2.2, lw=1.4,
                color=COLOR[cat], label=SHORT[cat], zorder=3)
        if band:
            ax.fill_between(solid.year, solid[f"{cat}_lo"], solid[f"{cat}_hi"],
                            color=COLOR[cat], alpha=0.10, lw=0, zorder=2)
        prov = sub[sub.year >= C.YEAR_END]
        if len(prov) > 1:
            ax.plot(prov.year, prov[col], ls=":", lw=1.1, color=COLOR[cat],
                    zorder=3)

    ax.axvspan(C.YEAR_END + 0.5, C.YEAR_END_DISPLAY + 0.4,
               color="0.88", alpha=0.6, lw=0, zorder=1)
    ax.set_xlabel("Filing year", fontsize=8)
    ax.set_ylabel(ylab, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, ls=":", lw=0.5)
    ax.set_xlim(start - 0.3, C.YEAR_END_DISPLAY + 0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", action="store_true")
    args = ap.parse_args()

    d = pd.read_csv(C.STAGE4 / "architecture_shares_filing_year.csv")
    d = d[(d.year >= C.YEAR_START) & (d.year <= C.YEAR_END_DISPLAY)]

    if args.single:
        fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.4))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2))

    # -- panel (a): counts. The honest picture: everything grows. ----------
    draw(axes[0], d, SHOW, "n", C.YEAR_START,
         "Families mentioning architecture")
    axes[0].set_title("(a) Counts", fontsize=8, loc="left")
    axes[0].legend(fontsize=6, frameon=False, ncol=2, loc="upper left",
                   handlelength=1.4, columnspacing=0.9, borderpad=0.2)

    # -- panel (b): shares, from SHARE_START only --------------------------
    draw(axes[1], d, SHOW, "pct", SHARE_START,
         "Share of families (%)", band=True)
    axes[1].set_title(f"(b) Shares, {SHARE_START} onward", fontsize=8,
                      loc="left")

    fig.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"fig_trajectory.{ext}", dpi=300,
                    bbox_inches="tight")
    print(f"wrote {FIGDIR}/fig_trajectory.{{pdf,png}}")

    # -- the numbers the caption should cite -------------------------------
    lo = d[d.year == 2017]
    hi = d[d.year == C.YEAR_END]
    print(f"\n  growth 2017 -> {C.YEAR_END}, counts:")
    tot_lo = float(lo.n_patents.iloc[0])
    tot_hi = float(hi.n_patents.iloc[0])
    print(f"    corpus            {tot_lo:>6.0f} -> {tot_hi:>6.0f}"
          f"  ({tot_hi/tot_lo:.2f}x)")
    for cat in SHOW:
        a = float(lo[f"{cat}_n"].iloc[0])
        b = float(hi[f"{cat}_n"].iloc[0])
        g = f"{b/a:.2f}x" if a else "n/a"
        print(f"    {SHORT[cat]:<17} {a:>6.0f} -> {b:>6.0f}  ({g})")
    print("\n  Cite these in the caption. A falling share with a rising count")
    print("  is composition change, not decline.")


if __name__ == "__main__":
    main()