#!/usr/bin/env python3
"""
STAGE 4 -- Architecture assignment, longitudinal shares, and the
naive-vs-validated corpus comparison.

Firm-level RTA is out of scope for this submission (see config.py). The
comparison against the unfiltered sweep replaces it as the empirical result.

Fixes carried over from the v1 audit:
  * word-boundary matching (v1 matched 'tpu' in "output", 'npu' in "input",
    'asic' in "basic" -- two of six series were noise)
  * filing year primary, publication year as the robustness run
  * Wilson intervals on every share
  * explicit statement that architecture classes are NOT mutually exclusive
  * explicit statement that architecture classes are NOT mutually exclusive

    python stage4_analysis.py
    python stage4_analysis.py --year-col pub_year     # robustness

Outputs:
    stage4_analysis/architecture_shares_<basis>.csv
    stage4_analysis/rta_by_firm.csv
    stage4_analysis/cluster_trends.csv
    stage4_analysis/validation_report.txt   (if labels are present)
"""

import argparse
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd

import config as C

PATTERNS = C.build_arch_patterns()


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (100 * max(0, c - h), 100 * min(1, c + h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year-col", default=C.YEAR_COL_PRIMARY)
    args = ap.parse_args()
    ycol = args.year_col

    print("=" * 72)
    print(f"STAGE 4 -- ANALYSIS (year basis: {ycol})")
    print("=" * 72)

    df = pd.read_parquet(C.STAGE3 / "clustered.parquet")
    print(f"  {len(df):,} records")

    text = df["text"].fillna("").astype(str)
    title = df["Title"].fillna("").astype(str)
    cats = list(PATTERNS)

    for cat, pat in PATTERNS.items():
        df[f"arch_{cat}"] = text.str.contains(pat, regex=True)
        df[f"archT_{cat}"] = title.str.contains(pat, regex=True)

    # ---- multi-label structure ------------------------------------------
    nlab = df[[f"arch_{c}" for c in cats]].sum(axis=1)
    print("\n  ARCHITECTURE LABEL STRUCTURE (classes are NOT exclusive)")
    print(f"    unlabelled      {100*(nlab == 0).mean():5.1f}%")
    print(f"    exactly one     {100*(nlab == 1).mean():5.1f}%")
    print(f"    two or more     {100*(nlab >= 2).mean():5.1f}%")
    print(f"    mean per patent {nlab.mean():.2f}")
    print("    -> shares sum above 100%. Label the axis a mention rate.")

    print("\n  EVIDENCE STRENGTH (title match = plausibly ABOUT the class)")
    for c in cats:
        a, t = int(df[f"arch_{c}"].sum()), int(df[f"archT_{c}"].sum())
        print(f"    {c:22} any {a:>8,}   title {t:>7,} "
              f"({100*t/a if a else 0:5.1f}%)")

    # ---- annual shares ---------------------------------------------------
    yr = pd.to_numeric(df[ycol], errors="coerce")
    rows = []
    for y in range(C.YEAR_START, C.YEAR_END_DISPLAY + 1):
        sub = df[yr == y]
        n = len(sub)
        r = {"year": y, "n_patents": n,
             "provisional": y > C.YEAR_END}
        for c in cats:
            k = int(sub[f"arch_{c}"].sum())
            lo, hi = wilson(k, n)
            r[f"{c}_n"] = k
            r[f"{c}_pct"] = 100 * k / n if n else np.nan
            r[f"{c}_lo"] = lo
            r[f"{c}_hi"] = hi
        rows.append(r)
    shares = pd.DataFrame(rows)
    shares.to_csv(C.STAGE4 / f"architecture_shares_{ycol}.csv", index=False)

    print(f"\n  SHARES BY {ycol.upper()} - % (count)")
    ms = [y for y in (2010, 2015, 2020, C.YEAR_END) if y in shares["year"].values]
    si = shares.set_index("year")
    hdr = "  " + f"{'architecture':<24}" + "".join(f"{y:>16}" for y in ms)
    print(hdr)
    for c in cats:
        cells = "".join(
            f"{si.loc[y, f'{c}_pct']:>9.2f} ({int(si.loc[y, f'{c}_n']):>4})"
            for y in ms)
        print(f"  {c:<24}{cells}")
    print(f"\n  Corpus size: " +
          ", ".join(f"{y}: {int(si.loc[y,'n_patents']):,}" for y in ms))
    print("  >> Early years are thin. In 2010 one patent shifts a share by")
    print("     ~1.7pp, so share movements there are not interpretable.")
    print(f"  Years after {C.YEAR_END} are right-truncated and flagged provisional.")

    # ---- cluster trends --------------------------------------------------
    if "tech_cluster" in df.columns:
        cl = df[df.tech_cluster != -1]
        ct = (cl.groupby(["tech_cluster", yr.loc[cl.index].rename("y")])
                .size().rename("n").reset_index())
        ct.to_csv(C.STAGE4 / "cluster_trends.csv", index=False)
        sizes = cl.tech_cluster.value_counts()
        print(f"\n  CLUSTERS: {len(sizes)} | noise "
              f"{100*(df.tech_cluster == -1).mean():.1f}% | "
              f"largest {100*sizes.max()/len(df):.1f}% of corpus")

    # ---- naive vs validated corpus --------------------------------------
    # The headline empirical result: what changes when the corpus is built
    # properly. Same architecture patterns, same year basis, two denominators.
    slim_path = C.STAGE1 / "sweep_slim.parquet"
    if slim_path.exists():
        print("\n" + "-" * 72)
        print("NAIVE SWEEP vs VALIDATED CORPUS")
        print("-" * 72)
        slim = pd.read_parquet(slim_path)
        stext = slim["text"].fillna("").astype(str)
        for c in cats:
            slim[f"arch_{c}"] = stext.str.contains(PATTERNS[c], regex=True)
        syr = pd.to_numeric(slim[ycol], errors="coerce")

        comp = []
        for y in range(C.YEAR_START, C.YEAR_END + 1):
            naive = slim[syr == y]
            valid = slim[(syr == y) & slim["in_corpus"]]
            if len(naive) == 0:
                continue
            row = {"year": y, "n_naive": len(naive), "n_validated": len(valid)}
            for c in cats:
                kn = int(naive[f"arch_{c}"].sum())
                kv = int(valid[f"arch_{c}"].sum()) if len(valid) else 0
                row[f"{c}_naive_n"] = kn
                row[f"{c}_valid_n"] = kv
                row[f"{c}_naive"] = 100 * kn / len(naive)
                row[f"{c}_valid"] = (100 * kv / len(valid)) if len(valid) else np.nan
                row[f"{c}_delta"] = row[f"{c}_valid"] - row[f"{c}_naive"]
            comp.append(row)
        cmp_df = pd.DataFrame(comp)
        cmp_df.to_csv(C.STAGE4 / f"naive_vs_validated_{ycol}.csv", index=False)

        # POOLED shares, not the mean of yearly shares. Averaging yearly
        # percentages gives 2010 (n=59) the same weight as 2021 (n=2,519).
        tot_n = cmp_df["n_naive"].sum()
        tot_v = cmp_df["n_validated"].sum()
        print(f"  Pooled over {C.YEAR_START}-{C.YEAR_END}: "
              f"naive n={tot_n:,}, validated n={tot_v:,}")
        print(f"  {'architecture':<24} {'naive %':>9} {'valid %':>9} "
              f"{'ratio':>8} {'valid n':>9}")
        summary = []
        for c in cats:
            kn, kv = cmp_df[f"{c}_naive_n"].sum(), cmp_df[f"{c}_valid_n"].sum()
            n = 100 * kn / tot_n
            v = 100 * kv / tot_v
            ratio = v / n if n else np.nan
            print(f"  {c:<24} {n:>9.3f} {v:>9.2f} {ratio:>7.1f}x {kv:>9,}")
            summary.append({"architecture": c, "naive_pooled_pct": n,
                            "validated_pooled_pct": v, "ratio": ratio,
                            "naive_n": int(kn), "validated_n": int(kv)})
        pd.DataFrame(summary).to_csv(
            C.STAGE4 / f"naive_vs_validated_summary_{ycol}.csv", index=False)

        # Trend comparison, restricted to years with enough mass to be stable.
        # In 2010 the corpus holds 59 patents, so one patent moves a share by
        # 1.7 points; sign flips there are noise, not findings.
        MIN_N = 300
        ok = cmp_df[cmp_df["n_validated"] >= MIN_N]
        print(f"\n  Trend direction, years with validated n >= {MIN_N} "
              f"({int(ok['year'].min()) if len(ok) else '-'}"
              f"-{int(ok['year'].max()) if len(ok) else '-'}):")
        print(f"  {'architecture':<24} {'naive':>9} {'validated':>11} "
              f"{'n first':>8} {'n last':>7}")
        k = max(len(ok) // 3, 1)
        for c in cats:
            dn = (ok[f"{c}_naive"].tail(k).mean() - ok[f"{c}_naive"].head(k).mean())
            dv = (ok[f"{c}_valid"].tail(k).mean() - ok[f"{c}_valid"].head(k).mean())
            n_first = int(ok[f"{c}_valid_n"].head(k).sum())
            n_last = int(ok[f"{c}_valid_n"].tail(k).sum())
            flag = "  OPPOSITE" if np.sign(dn) != np.sign(dv) else ""
            print(f"  {c:<24} {dn:>+8.2f}pp {dv:>+10.2f}pp "
                  f"{n_first:>8,} {n_last:>7,}{flag}")
        print("\n  >> Read shares and counts together. A falling SHARE with a")
        print("     rising COUNT is a composition effect: the corpus grew and")
        print("     newer architectures entered, not that the technology")
        print("     declined. Report both, and never a share trend alone.")

    # ---- validation report if labels exist ------------------------------
    vp = C.STAGE1 / "validation_sample_labelled.csv"
    if vp.exists():
        val = pd.read_csv(vp)
        for c in ("is_ai_hardware", "architecture", "is_about_class", "_side"):
            if c in val.columns:
                val[c] = val[c].fillna("").astype(str).str.strip()
            else:
                val[c] = ""
        lab = val[val["is_ai_hardware"] != ""]
        if len(lab) == 0:
            print(f"\n  {vp.name} exists but has no labels yet "
                  f"({len(val)} rows awaiting annotation).")
        else:
            ret = lab[lab["_side"] == "retained"]
            near = lab[lab["_side"] == "excluded_near"]
            yes = lambda s: (s.str.upper().str[:1] == "Y").mean() if len(s) else np.nan
            prec = yes(ret["is_ai_hardware"])
            fn = yes(near["is_ai_hardware"])
            about = yes(lab.loc[lab["is_about_class"] != "", "is_about_class"])
            lines = [
                f"Labelled items: {len(lab)} of {len(val)}",
                f"Retained judged AI hardware (precision): "
                f"{100*prec:.1f}%  (n={len(ret)})",
            ]
            if len(near):
                lines.append(f"Near-miss exclusions judged AI hardware "
                             f"(false negatives): {100*fn:.1f}%  (n={len(near)})")
            if not np.isnan(about):
                lines.append(f"Labelled 'about' the class rather than mentioning "
                             f"it: {100*about:.1f}%")
            (C.STAGE4 / "validation_report.txt").write_text("\n".join(lines))
            print("\n  VALIDATION")
            for l in lines:
                print("    " + l)
    else:
        print(f"\n  No labelled validation file at {vp}")
        print("    Label validation_sample.csv and save it with that name.")

    (C.STAGE4 / "manifest.json").write_text(json.dumps({
        "stage": 4, "timestamp": datetime.now().isoformat(),
        "year_basis": ycol, "n_records": len(df),
        "architectures": cats,
        "mean_labels_per_patent": float(nlab.mean()),
        "unlabelled_pct": float(100 * (nlab == 0).mean()),
    }, indent=2))
    print(f"\n  Wrote -> {C.STAGE4}")
    print("=" * 72)


if __name__ == "__main__":
    main()