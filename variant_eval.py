#!/usr/bin/env python3
"""
Score candidate corpus rules against the human-labelled sample.

T2 precision came out at 7.1% (1/14). This evaluates rule variants against the
labels already collected, so the fix can be chosen from evidence rather than
guessed at.

    python variant_eval.py

IMPORTANT: choosing a rule on these labels and then reporting precision on the
SAME labels is optimistic. Use this to diagnose and select, then label the
remaining rows on the NEW sample as a clean held-out estimate. Say so in the
paper.
"""

import ast

import numpy as np
import pandas as pd

import config as C


def parse_cpc(cell):
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if s.startswith("["):
        try:
            return sorted(set(ast.literal_eval(s)))
        except Exception:
            pass
    return sorted({c.strip() for c in s.split(";;") if c.strip()})


def any_prefix(codes, prefixes):
    return any(c.startswith(p) for c in codes for p in prefixes)


def norm(s):
    s = str(s).strip().upper()
    return s[:1] if s and s != "NAN" else ""


# G06N subgroups that denote HARDWARE realisation rather than models/methods.
# G06N3/08 (learning), G06N20 (ML generally), G06N5 (knowledge-based) and
# G06N7 (probabilistic) are software and should not count as AI-hardware
# evidence on their own.
AI_CPC_HW = ("G06N3/06",)
AI_CPC_ANY = ("G06N",)


def build(df):
    codes = df["_cpc"]
    text = df["_text"]
    e = pd.DataFrame(index=df.index)
    e["t1"] = codes.map(lambda L: any_prefix(L, C.TIER1_CPC_PREFIXES))
    e["ai_cpc_any"] = codes.map(lambda L: any_prefix(L, AI_CPC_ANY)
                                and not any_prefix(L, C.AI_CPC_EXCLUDE))
    e["ai_cpc_hw"] = codes.map(lambda L: any_prefix(L, AI_CPC_HW))
    e["hw_cpc"] = codes.map(lambda L: any_prefix(L, C.HW_CPC_PREFIXES))
    e["ai_txt"] = text.str.contains(C.AI_TEXT_RE, regex=True)
    e["ai_mech"] = text.str.contains(C.AI_MECHANISM_RE, regex=True)
    e["hw_txt"] = text.str.contains(C.HW_TEXT_STRONG_RE, regex=True)
    e["veto"] = df["_title"].str.contains(C.EXCLUDE_SUBJECT_RE, regex=True)
    return e


VARIANTS = {
    "V0 current (AI cpc|txt|mech) & (HW cpc|txt)":
        lambda e: e.t1 | ((e.ai_cpc_any | e.ai_txt | e.ai_mech)
                          & (e.hw_cpc | e.hw_txt)),
    "V1 T2 needs strong HW TEXT":
        lambda e: e.t1 | ((e.ai_cpc_any | e.ai_txt | e.ai_mech) & e.hw_txt),
    "V2 T2 needs AI TEXT/mech":
        lambda e: e.t1 | ((e.ai_txt | e.ai_mech) & (e.hw_cpc | e.hw_txt)),
    "V3 T2 needs BOTH texts":
        lambda e: e.t1 | ((e.ai_txt | e.ai_mech) & e.hw_txt),
    "V4 T1 only (G06N3/06)":
        lambda e: e.t1,
    # --- text-gated T1: every FP above met both conditions on CPC alone -----
    "V7 T1 gated on ANY text evidence":
        lambda e: e.t1 & (e.hw_txt | e.ai_txt | e.ai_mech),
    "V8 T1 gated on HW text":
        lambda e: e.t1 & e.hw_txt,
    "V9 HW text required, AI from T1 or text":
        lambda e: e.hw_txt & (e.t1 | e.ai_txt | e.ai_mech),
    "V10 V8 or both-texts T2":
        lambda e: (e.t1 & e.hw_txt) | ((e.ai_txt | e.ai_mech) & e.hw_txt),
    "V11 T1 gated on HW evidence (txt or cpc)":
        lambda e: e.t1 & (e.hw_txt | e.hw_cpc),
}


def main():
    lab = pd.read_csv(C.STAGE1 / "validation_sample_labelled.csv")
    lab["gold"] = lab["is_ai_hardware"].map(norm)
    lab = lab[lab["gold"].isin(["Y", "N", "B"])].copy()
    if not len(lab):
        raise SystemExit("No labels found in validation_sample_labelled.csv")

    cache = pd.read_parquet(C.STAGE1 / "ingest_cache.parquet")
    cpc_col = next(c for c in cache.columns if "CPC" in c)
    cache = cache[[C.ID_COL, "Title", "Abstract", cpc_col]]
    d = lab[[C.ID_COL, "gold", "_side", "tier"]].merge(cache, on=C.ID_COL,
                                                       how="left")
    d["_cpc"] = d[cpc_col].map(parse_cpc)
    d["_title"] = d["Title"].fillna("").astype(str)
    d["_text"] = (d["_title"] + ". " + d["Abstract"].fillna("").astype(str))

    e = build(d)
    gold_y = d["gold"] == "Y"
    n_pos = int(gold_y.sum())

    print("=" * 78)
    print(f"RULE VARIANTS vs {len(d)} HUMAN LABELS "
          f"({n_pos} judged Y, {(d.gold=='N').sum()} N, "
          f"{(d.gold=='B').sum()} BORDERLINE)")
    print("  BORDERLINE counted as NOT AI hardware (conservative)")
    print("=" * 78)
    print(f"  {'variant':<48} {'kept':>5} {'TP':>4} {'FP':>4} {'FN':>4} "
          f"{'prec':>6} {'rec':>6} {'F1':>6}")
    rows = []
    for name, fn in VARIANTS.items():
        keep = fn(e) & ~e.veto
        tp = int((keep & gold_y).sum())
        fp = int((keep & ~gold_y).sum())
        fn_ = int((~keep & gold_y).sum())
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / n_pos if n_pos else np.nan
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else np.nan
        print(f"  {name:<48} {int(keep.sum()):>5} {tp:>4} {fp:>4} {fn_:>4} "
              f"{100*prec:>5.1f}% {100*rec:>5.1f}% {100*f1:>5.1f}%")
        rows.append(dict(variant=name, kept=int(keep.sum()), tp=tp, fp=fp,
                         fn=fn_, precision=prec, recall=rec, f1=f1))
    pd.DataFrame(rows).to_csv(C.STAGE1 / "variant_eval.csv", index=False)

    print("\n  FALSE NEGATIVES (judged Y but NOT kept by the current rule):")
    v0 = VARIANTS["V0 current (AI cpc|txt|mech) & (HW cpc|txt)"](e) & ~e.veto
    miss = d[gold_y & ~v0]
    if not len(miss):
        print("    none")
    for i, r in miss.iterrows():
        ev = [k for k in ("ai_cpc_any", "ai_txt", "ai_mech", "hw_cpc", "hw_txt")
              if e.loc[i, k]]
        print(f"    [{r['_side']}] {str(r['Title'])[:56]:58} "
              f"has {','.join(ev) or 'no evidence'}")

    print("\n  FALSE POSITIVES UNDER V0 (what the current rule wrongly keeps):")
    bad = d[v0 & ~gold_y]
    for i, r in bad.head(12).iterrows():
        ev = [k for k in ("ai_cpc_any", "ai_txt", "ai_mech", "hw_cpc", "hw_txt")
              if e.loc[i, k]]
        print(f"    [{r['tier']}] {str(r['Title'])[:56]:58} via {','.join(ev)}")

    print("\n  CAUTION: 50 labels and 10 variants means selection noise is real.")
    print("  Differences of a few points are not meaningful at this n. Prefer")
    print("  the variant whose LOGIC matches the diagnosed failure, then confirm")
    print("  on the held-out rows rather than picking the maximum F1.")

    print("\n  Choose on F1, but weight precision: a corpus that is 40% wrong")
    print("  cannot support the paper's central claim. Then label the REMAINING")
    print("  rows on the new sample for a clean held-out estimate.")
    print(f"\n  Wrote {C.STAGE1 / 'variant_eval.csv'}")


if __name__ == "__main__":
    main()