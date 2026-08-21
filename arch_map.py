#!/usr/bin/env python3
"""
Map free-text architecture labels onto the taxonomy, then score the keyword
assignment against them.

The human labels were written as descriptions ("Systolic array with MAC
processing elements") rather than category codes. Mapping them gives a
confusion matrix for the architecture assignment -- which is what reviewer
comment W1 asked for and what no version of this analysis has had.

    python arch_map.py template   # writes arch_mapping.csv for you to fill
    python arch_map.py score      # confusion matrix + per-class precision

Fill the `category` column in arch_mapping.csv with one of:
    CPU  GPU  TPU-NPU  FPGA-ASIC  NEUROMORPHIC  PIM  DPU-VPU-DSP  ANALOG  NONE
Comma-separate if a description spans two.
"""

import sys

import numpy as np
import pandas as pd

import config as C

import os
# Annotator file can be overridden: ARCH_FILE=validation_blind_human2_full.csv
BLIND = C.STAGE1 / os.environ.get("ARCH_FILE",
                                  "validation_blind_human2_full.csv")
MAP = C.STAGE1 / "arch_mapping.csv"

# Rows 1-120 were labelled BEFORE the vocabulary was revised, so they are the
# development set; rows 121+ were labelled after the rule was frozen and give a
# held-out estimate. Reporting them separately is the honest split.
DEV_MAX_ROW = 120

CANON = {
    "CPU": "CPU", "GPU": "GPU", "TPU-NPU": "TPU/NPU", "TPU/NPU": "TPU/NPU",
    "FPGA-ASIC": "FPGA/ASIC", "FPGA/ASIC": "FPGA/ASIC",
    "NEUROMORPHIC": "Neuromorphic/Memristive",
    "PIM": "PIM/Memory-Centric",
    "DPU-VPU-DSP": "DPU/VPU/DSP", "DPU/VPU/DSP": "DPU/VPU/DSP",
    "ANALOG": "Analog/Photonic", "NONE": "NONE",
}

# Suggestions to pre-fill the template; you still review every row.
HINTS = [
    ("systolic|mac |multiply.accumul|tensor core|dataflow|accelerat", "TPU-NPU"),
    ("neuromorph|memrist|synap|spiking|neuron|crossbar", "NEUROMORPHIC"),
    ("in.memory|processing.in.memory|pim\\b|near.memory|computational memory",
     "PIM"),
    ("fpga|asic|reconfigurable|programmable logic", "FPGA-ASIC"),
    ("gpu|graphics", "GPU"),
    ("photonic|optical|analog", "ANALOG"),
    ("dsp|vision processing|digital signal", "DPU-VPU-DSP"),
    ("cpu|central processing|general.purpose (host|processor|microprocessor)",
     "CPU"),
]


def read_any(p):
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(p, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(p, encoding="latin-1")


def template():
    d = read_any(BLIND)
    a = d["architecture"].fillna("").astype(str).str.strip()
    vals = sorted(set(a[a != ""]))
    rows = []
    for v in vals:
        sug = ""
        low = v.lower()
        for pat, cat in HINTS:
            import re
            if re.search(pat, low):
                sug = cat
                break
        if v.upper() == "NONE":
            sug = "NONE"
        rows.append({"free_text": v, "suggested": sug, "category": sug,
                     "n": int((a == v).sum())})
    out = pd.DataFrame(rows).sort_values("n", ascending=False)
    out.to_csv(MAP, index=False, encoding="utf-8")
    print(f"  Wrote {MAP} ({len(out)} distinct descriptions)")
    print(f"  {int((out.category == '').sum())} rows need a category filled in.")
    print("  Review EVERY suggested value -- they are keyword guesses, not"
          " judgements.")


def score():
    d = read_any(BLIND)
    m = read_any(MAP)
    m["category"] = m["category"].fillna("").astype(str).str.strip()
    lut = dict(zip(m.free_text.astype(str), m.category))

    d["arch_free"] = d["architecture"].fillna("").astype(str).str.strip()
    d = d[d.arch_free != ""].copy()
    d["gold"] = d.arch_free.map(lambda v: set(
        CANON.get(c.strip().upper(), c.strip())
        for c in str(lut.get(v, "")).split(",") if c.strip()))
    d = d[d.gold.map(len) > 0]
    if not len(d):
        sys.exit("  No mapped categories. Fill the `category` column first.")

    corpus = pd.read_parquet(C.STAGE1 / "corpus.parquet",
                             columns=[C.ID_COL, "text"])
    d = d.merge(corpus, on=C.ID_COL, how="inner")
    print(f"  {len(d)} labelled records matched to the corpus")

    P = C.build_arch_patterns()
    cats = list(P)
    for c in cats:
        d[f"kw_{c}"] = d["text"].fillna("").str.contains(P[c], regex=True)

    if "row" in d.columns:
        dev = d[d.row <= DEV_MAX_ROW]
        held = d[d.row > DEV_MAX_ROW]
        print(f"  development set (rows 1-{DEV_MAX_ROW}): {len(dev)}")
        print(f"  held-out set (rows {DEV_MAX_ROW+1}+):    {len(held)}")
        splits = [("ALL", d), ("DEVELOPMENT (vocabulary tuned on these)", dev),
                  ("HELD-OUT (vocabulary frozen)", held)]
    else:
        splits = [("ALL", d)]
    print()

    for label, sub in splits:
        if not len(sub):
            continue
        print("-" * 74)
        print(f"  {label}  (n={len(sub)})")
        print("-" * 74)
        report(sub, cats, label)


def report(d, cats, label):
    import numpy as np
    print(f"  {'architecture':<26} {'gold':>5} {'kw':>5} {'TP':>4} "
          f"{'FP':>4} {'FN':>4} {'prec':>7} {'rec':>7}")
    rows = []
    import pandas as pd
    for c in cats:
        gold = d.gold.map(lambda s: c in s)
        kw = d[f"kw_{c}"]
        tp = int((gold & kw).sum())
        fp = int((~gold & kw).sum())
        fn = int((gold & ~kw).sum())
        pr = tp / (tp + fp) if tp + fp else np.nan
        rc = tp / (tp + fn) if tp + fn else np.nan
        print(f"  {c:<26} {int(gold.sum()):>5} {int(kw.sum()):>5} {tp:>4} "
              f"{fp:>4} {fn:>4} "
              f"{100*pr if pr==pr else float('nan'):>6.1f}% "
              f"{100*rc if rc==rc else float('nan'):>6.1f}%")
        rows.append(dict(architecture=c, n_gold=int(gold.sum()),
                         n_keyword=int(kw.sum()), tp=tp, fp=fp, fn=fn,
                         precision=pr, recall=rc))
    tag = label.split()[0].lower()
    pd.DataFrame(rows).to_csv(
        C.STAGE4 / f"architecture_validation_{tag}.csv", index=False)

    none_gold = d.gold.map(lambda s: s == {"NONE"})
    any_kw = d[[f"kw_{c}" for c in cats]].any(axis=1)
    print(f"\n  Human said NONE, keyword assigned something: "
          f"{int((none_gold & any_kw).sum())} of {int(none_gold.sum())}")
    print(f"  Human named a class, keyword found nothing: "
          f"{int((~none_gold & ~any_kw).sum())} of {int((~none_gold).sum())}")
    print(f"    -> architecture_validation_{tag}.csv")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "template"
    {"template": template, "score": score}.get(cmd, template)()