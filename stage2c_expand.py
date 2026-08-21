#!/usr/bin/env python3
"""
STAGE 2C -- Semantic corpus expansion from a validated seed set.

The lexical rule (V7) is precise but CPC-anchored: a patent whose abstract
plainly describes a neural network accelerator is missed if the examiner never
assigned G06N3/06. Semantic retrieval recovers those without reintroducing the
keyword noise that made Tier 2 unusable.

Method
------
1. Seeds = the retained corpus (validated, ~80% precision).
2. For every non-seed candidate, score = mean cosine similarity to its k
   nearest seeds. Mean-of-top-k rather than max, so one atypical seed cannot
   drag in a neighbourhood on its own.
3. Threshold chosen by labelling a stratified sample across score bands and
   finding where precision falls below the seed corpus's own precision.

    python stage2c_expand.py score              # compute scores
    python stage2c_expand.py sample             # stratified labelling sample
    python stage2c_expand.py apply --threshold 0.82

Outputs (stage1_corpus/):
    expansion_scores.parquet       Lens ID, score, is_seed
    expansion_label_sample.csv     stratified by band -- LABEL THIS
    corpus_expanded.parquet        seeds + accepted candidates
"""

import argparse
import sys

import numpy as np
import pandas as pd

import config as C

SCORES = C.STAGE1 / "expansion_scores.parquet"
SAMPLE = C.STAGE1 / "expansion_label_sample.csv"
EXPANDED = C.STAGE1 / "corpus_expanded.parquet"
BANDS = [(0.90, 1.01), (0.85, 0.90), (0.80, 0.85),
         (0.75, 0.80), (0.70, 0.75), (0.60, 0.70)]


def load_all():
    emb = np.load(C.STAGE2 / "sweep_embeddings.npy", mmap_mode="r")
    ids = pd.read_csv(C.STAGE2 / "sweep_ids.csv")[C.ID_COL].astype(str)
    if len(ids) != emb.shape[0]:
        sys.exit(f"FATAL: {len(ids)} ids vs {emb.shape[0]} embedding rows")
    corpus = pd.read_parquet(C.STAGE1 / "corpus.parquet", columns=[C.ID_COL])
    seed_ids = set(corpus[C.ID_COL].astype(str))
    is_seed = ids.isin(seed_ids).values
    print(f"  {emb.shape[0]:,} embedded, {is_seed.sum():,} seeds")
    return emb, ids, is_seed


def score(k=10, chunk=20_000):
    emb, ids, is_seed = load_all()
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        sys.exit("torch required")

    S = torch.tensor(np.asarray(emb[is_seed]), device=dev)      # [n_seed, dim]
    print(f"  Seed matrix {tuple(S.shape)} on {dev}; scoring with k={k}")

    n = emb.shape[0]
    out = np.zeros(n, dtype=np.float32)
    seed_flag = torch.tensor(is_seed, device=dev)
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        Q = torch.tensor(np.asarray(emb[i:j]), device=dev)
        sim = Q @ S.T                                            # cosine
        # Leave-one-out for seeds: a seed matches itself at 1.0, which would
        # inflate the reference scale the threshold is judged against.
        kk = min(k + 1, S.shape[0])
        top = torch.topk(sim, kk, dim=1).values
        is_s = seed_flag[i:j].unsqueeze(1)
        # for seeds drop the first (self) column, for candidates drop the last
        loo = torch.where(is_s, top[:, 1:kk], top[:, 0:kk - 1])
        out[i:j] = loo.mean(dim=1).cpu().numpy()
        del Q, sim, top, loo
        if (i // chunk) % 10 == 0:
            print(f"    {j:,}/{n:,}", flush=True)

    df = pd.DataFrame({C.ID_COL: ids.values, "score": out, "is_seed": is_seed})
    df.to_parquet(SCORES, index=False)

    cand = df[~df.is_seed]
    seeds = df[df.is_seed]
    print(f"\n  Candidate score distribution (n={len(cand):,}):")
    for lo, hi in BANDS:
        m = (cand.score >= lo) & (cand.score < hi)
        print(f"    {lo:.2f}-{hi:.2f}: {int(m.sum()):>9,}")
    print(f"    below 0.60 : {int((cand.score < 0.60).sum()):>9,}")

    print(f"\n  Seed reference scale (leave-one-out):")
    for q in (0.10, 0.25, 0.50, 0.75, 0.90):
        print(f"    p{int(q*100):>2}: {seeds.score.quantile(q):.3f}")
    print(f"\n  Candidates scoring above the seed median "
          f"({seeds.score.median():.3f}): "
          f"{int((cand.score > seeds.score.median()).sum()):,}")
    print("  >> A diffuse seed neighbourhood means the threshold cannot be read")
    print("     off the distribution; it must come from the labelled bands.")
    print(f"  Wrote {SCORES}")


def sample(per_band=20):
    sc = pd.read_parquet(SCORES)
    cache = pd.read_parquet(C.STAGE1 / "ingest_cache.parquet",
                            columns=[C.ID_COL, "Display Key", "Title",
                                     "Abstract"])
    cand = sc[~sc.is_seed].merge(cache, on=C.ID_COL, how="left")
    rows = []
    for lo, hi in BANDS:
        g = cand[(cand.score >= lo) & (cand.score < hi)]
        if len(g):
            rows.append(g.sample(min(per_band, len(g)), random_state=C.SEED)
                        .assign(band=f"{lo:.2f}-{hi:.2f}"))
    out = pd.concat(rows)
    out["is_ai_hardware"] = ""
    out["architecture"] = ""
    out["notes"] = ""
    out = out.sample(frac=1, random_state=C.SEED).reset_index(drop=True)
    out.insert(0, "row", range(1, len(out) + 1))
    keep = ["row", C.ID_COL, "Display Key", "Title", "Abstract", "score",
            "band", "is_ai_hardware", "architecture", "notes"]
    out[keep].to_csv(SAMPLE, index=False)
    print(f"  Wrote {SAMPLE} ({len(out)} rows, {per_band} per band)")
    print("  Label is_ai_hardware (Y/N/BORDERLINE), then:")
    print("    python stage2c_expand.py calibrate")


def calibrate():
    d = pd.read_csv(SAMPLE)
    d["lab"] = d["is_ai_hardware"].fillna("").astype(str).str.strip().str.upper().str[:1]
    d = d[d.lab.isin(["Y", "N", "B"])]
    if not len(d):
        sys.exit("  No labels in expansion_label_sample.csv yet.")
    print(f"  Labelled {len(d)} candidates\n")
    print(f"  {'band':>12} {'n':>4} {'Y':>4} {'precision':>10}")
    for b, g in d.groupby("band", sort=True):
        y = int((g.lab == "Y").sum())
        print(f"  {b:>12} {len(g):>4} {y:>4} {100*y/len(g):>9.1f}%")
    print("\n  Cumulative precision if threshold set at each band floor:")
    for lo, hi in BANDS:
        g = d[d.score >= lo]
        if len(g) >= 5:
            y = int((g.lab == "Y").sum())
            print(f"    >= {lo:.2f}: {y}/{len(g)} = {100*y/len(g):>5.1f}%")
    print("\n  Pick the lowest threshold whose cumulative precision still")
    print("  matches the seed corpus (~80%). Below that, expansion dilutes")
    print("  the corpus rather than extending it.")


def apply(threshold):
    sc = pd.read_parquet(SCORES)
    corpus = pd.read_parquet(C.STAGE1 / "corpus.parquet")
    cache = pd.read_parquet(C.STAGE1 / "ingest_cache.parquet")

    acc = sc[(~sc.is_seed) & (sc.score >= threshold)][[C.ID_COL, "score"]]
    print(f"  Threshold {threshold}: {len(acc):,} candidates accepted")

    add = cache.merge(acc, on=C.ID_COL, how="inner")
    # the subject veto still applies to expanded records
    veto = add["Title"].fillna("").astype(str).str.contains(
        C.EXCLUDE_SUBJECT_RE, regex=True)
    print(f"    {int(veto.sum()):,} removed by subject veto")
    add = add[~veto].copy()
    add["source"] = "expanded"
    add["tier"] = "EXP"

    corpus = corpus.copy()
    corpus["source"] = "lexical"
    if "score" not in corpus.columns:
        corpus["score"] = np.nan

    common = [c for c in corpus.columns if c in add.columns]
    out = pd.concat([corpus[common], add[common]], ignore_index=True)
    out = out.drop_duplicates(C.ID_COL)
    out.to_parquet(EXPANDED, index=False)
    print(f"\n  Corpus: {len(corpus):,} lexical + {len(add):,} expanded "
          f"= {len(out):,}")
    print(f"  Wrote {EXPANDED}")
    print("  Re-run Stage 2 and Stage 3 against this file to cluster the")
    print("  expanded corpus. Report lexical and expanded precision separately.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["score", "sample", "calibrate", "apply"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--per-band", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.85)
    a = ap.parse_args()
    if a.cmd == "score":
        score(k=a.k)
    elif a.cmd == "sample":
        sample(per_band=a.per_band)
    elif a.cmd == "calibrate":
        calibrate()
    else:
        apply(a.threshold)