#!/usr/bin/env python3
"""
Blind labelling with named annotators, plus human-vs-LLM agreement.

    python blind_labels2.py make human llm     # create one blind file each
    python blind_labels2.py merge human        # human labels -> Stage 4 input
    python blind_labels2.py agree human llm    # Cohen's kappa + disagreements

Files (all under stage1_corpus/):
    validation_sample.csv            source, written by Stage 1. Never edit.
    validation_blind_<name>.csv      the file each annotator fills in.
    validation_sample_labelled.csv   merge output; Stage 4 reads this.
    validation_agreement.csv         disagreement table from `agree`.

The human labels are the gold standard. The LLM is a comparison condition --
never average the two, and never present them as two interchangeable
annotators in the paper.
"""

import sys

import numpy as np
import pandas as pd

import config as C


def read_csv_safe(path, **kw):
    """Read a CSV that may have been round-tripped through Excel.

    Excel writes Windows-1252 by default, so an en dash becomes byte 0x96 and
    utf-8 decoding fails. Try utf-8 first, then the usual Windows encodings.
    """
    import pandas as _pd
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return _pd.read_csv(path, encoding=enc, **kw)
        except UnicodeDecodeError:
            continue
    return _pd.read_csv(path, encoding="latin-1", errors="replace", **kw)

SRC = C.STAGE1 / "validation_sample.csv"
OUT = C.STAGE1 / "validation_sample_labelled.csv"

HIDE = ["tier", "_side", "_source", "in_corpus", "cond_ai", "cond_hw"]
LABEL_COLS = ["is_ai_hardware", "architecture", "is_about_class", "notes"]


def blind_path(name):
    return C.STAGE1 / f"validation_blind_{name}.csv"


def norm(s):
    """Normalise a label to Y / N / B / '' so typos don't count as disagreement."""
    s = str(s).strip().upper()
    if not s or s == "NAN":
        return ""
    if s.startswith("Y"):
        return "Y"
    if s.startswith("N"):
        return "N"
    if s.startswith("B"):
        return "B"
    return s[:1]


def make(names):
    df = read_csv_safe(SRC)
    keep = [c for c in df.columns if c not in HIDE and c not in LABEL_COLS]
    base = df[keep].copy()
    for c in LABEL_COLS:
        base[c] = ""
    # Same shuffle for every annotator, so `row` refers to the same patent.
    base = base.sample(frac=1, random_state=C.SEED).reset_index(drop=True)
    base.insert(0, "row", range(1, len(base) + 1))

    for name in names:
        p = blind_path(name)
        if p.exists():
            old = read_csv_safe(p)
            done = (old.get("is_ai_hardware", pd.Series(dtype=str))
                    .fillna("").astype(str).str.strip() != "").sum()
            if done:
                print(f"  {p.name}: {done} rows already labelled -- not touching it")
                continue
        base.to_csv(p, index=False)
        print(f"  Wrote {p.name} ({len(base)} rows)")
    print("\n  Row numbers are identical across annotator files, so row 7 is the")
    print("  same patent for everyone. Label independently; do not compare until")
    print("  both are finished.")


def merge(name):
    p = blind_path(name)
    if not p.exists():
        sys.exit(f"  {p} not found. Run: python blind_labels2.py make {name}")
    blind = read_csv_safe(p)
    src = read_csv_safe(SRC)
    for c in LABEL_COLS:
        if c not in blind.columns:
            blind[c] = ""
        blind[c] = blind[c].fillna("").astype(str).str.strip()

    merged = src.drop(columns=[c for c in LABEL_COLS if c in src.columns],
                      errors="ignore").merge(
        blind[[C.ID_COL] + LABEL_COLS], on=C.ID_COL, how="left")
    for c in LABEL_COLS:
        merged[c] = merged[c].fillna("").astype(str).str.strip()
    merged.to_csv(OUT, index=False)

    done = merged[merged["is_ai_hardware"] != ""].copy()
    print(f"  Wrote {OUT.name} from {p.name}")
    print(f"  Labelled: {len(done)} of {len(merged)}")
    if not len(done):
        return

    done["lab"] = done["is_ai_hardware"].map(norm)
    print("\n  BY SIDE (revealed only now):")
    print("    " + pd.crosstab(done["_side"], done["lab"])
          .to_string().replace("\n", "\n    "))

    ret = done[done["_side"] == "retained"]
    if len(ret):
        y = (ret["lab"] == "Y").sum()
        b = (ret["lab"] == "B").sum()
        n = len(ret)
        lo, hi = wilson(y, n)
        print(f"\n  PRECISION (retained judged Y): {y}/{n} = {100*y/n:.1f}% "
              f"[95% CI {lo:.1f}-{hi:.1f}]")
        if b:
            print(f"    plus {b} BORDERLINE ({100*b/n:.1f}%); upper bound if all "
                  f"counted Y: {100*(y+b)/n:.1f}%")

    # Tier 1 is 61% of the corpus, admitted with NO text evidence. Its
    # precision is the single most important number in this validation.
    if "tier" in ret.columns and len(ret):
        print("\n  PRECISION BY TIER:")
        for t, g in ret.groupby("tier"):
            y = (g["lab"] == "Y").sum()
            lo, hi = wilson(y, len(g))
            print(f"    {t}: {y}/{len(g)} = {100*y/len(g):.1f}% "
                  f"[{lo:.1f}-{hi:.1f}]")

    near = done[done["_side"] == "excluded_near"]
    if len(near):
        y = (near["lab"] == "Y").sum()
        print(f"\n  FALSE NEGATIVES (near-miss exclusions judged Y): "
              f"{y}/{len(near)} = {100*y/len(near):.1f}%")

    rem = len(merged) - len(done)
    if rem:
        print(f"\n  {rem} rows still blank.")


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (100 * max(0, c - h), 100 * min(1, c + h))


def kappa(a, b):
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = np.mean([x == y for x, y in zip(a, b)])
    pe = sum((np.mean([x == c for x in a]) * np.mean([y == c for y in b]))
             for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan, po


def agree(n1, n2):
    d1 = read_csv_safe(blind_path(n1))
    d2 = read_csv_safe(blind_path(n2))
    m = d1[["row", C.ID_COL, "Title", "is_ai_hardware", "architecture"]].merge(
        d2[["row", "is_ai_hardware", "architecture"]], on="row",
        suffixes=(f"_{n1}", f"_{n2}"))
    m["a"] = m[f"is_ai_hardware_{n1}"].map(norm)
    m["b"] = m[f"is_ai_hardware_{n2}"].map(norm)
    both = m[(m["a"] != "") & (m["b"] != "")]
    if not len(both):
        sys.exit("  No rows labelled by both annotators yet.")

    k, po = kappa(both["a"].tolist(), both["b"].tolist())
    print(f"  Rows labelled by both: {len(both)}")
    print(f"  Raw agreement:  {100*po:.1f}%")
    print(f"  Cohen's kappa:  {k:.3f}  ({interpret(k)})")
    print("\n  CONFUSION (rows = {}, cols = {}):".format(n1, n2))
    print("    " + pd.crosstab(both["a"], both["b"])
          .to_string().replace("\n", "\n    "))

    dis = both[both["a"] != both["b"]]
    print(f"\n  DISAGREEMENTS: {len(dis)}")
    for _, r in dis.head(25).iterrows():
        print(f"    row {r['row']:>3} [{n1}={r['a']} {n2}={r['b']}] "
              f"{str(r['Title'])[:64]}")
    out = C.STAGE1 / "validation_agreement.csv"
    dis.to_csv(out, index=False)
    print(f"\n  Wrote {out.name}")
    print("  Adjudicate these by hand. The disagreements ARE the analysis --")
    print("  they show where the AI-hardware boundary is genuinely contested.")


def interpret(k):
    if np.isnan(k):
        return "undefined"
    for t, s in ((0.81, "almost perfect"), (0.61, "substantial"),
                 (0.41, "moderate"), (0.21, "fair"), (0.0, "slight")):
        if k >= t:
            return s
    return "poor"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "make":
        make(args or ["human", "llm"])
    elif cmd == "merge":
        merge(args[0] if args else "human")
    elif cmd == "agree":
        agree(*(args[:2] if len(args) >= 2 else ["human", "llm"]))
    else:
        sys.exit(__doc__)