#!/usr/bin/env python3
"""
STAGE 2B -- Embed the FULL sweep, not just the retained corpus.

Semantic expansion needs vectors for every candidate, including the 1.47M the
lexical rule rejected. Same model and normalisation as Stage 2, so seed and
candidate vectors are directly comparable.

Writes a float32 memmap plus an ID index, chunked and resumable -- an
interrupted run picks up where it stopped.

    python stage2b_embed_sweep.py
    python stage2b_embed_sweep.py --limit 50000   # smoke test

Outputs (stage2_embeddings/):
    sweep_embeddings.npy      float32 [N, dim], unit-normalised
    sweep_ids.csv             one Lens ID per row, same order
    sweep_embed_manifest.json
"""

import argparse
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

import config as C

OUT_EMB = C.STAGE2 / "sweep_embeddings.npy"
OUT_IDS = C.STAGE2 / "sweep_ids.csv"
OUT_MAN = C.STAGE2 / "sweep_embed_manifest.json"
PROGRESS = C.STAGE2 / "sweep_embed_progress.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=50_000)
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print("STAGE 2B -- EMBED FULL SWEEP")
    print("=" * 72)

    slim = pd.read_parquet(C.STAGE1 / "sweep_slim.parquet",
                           columns=[C.ID_COL, "text", "in_corpus"])
    if args.limit:
        slim = slim.head(args.limit)
    n = len(slim)
    texts = slim["text"].fillna("").astype(str).tolist()
    print(f"  Candidates: {n:,}  (of which {int(slim.in_corpus.sum()):,} "
          f"are current corpus members)")

    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = C.EMBED_MODEL
    try:
        model = SentenceTransformer(model_name, device=device)
    except Exception as e:
        print(f"  {model_name} failed ({e}); using {C.EMBED_FALLBACK}")
        model_name = C.EMBED_FALLBACK
        model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = C.EMBED_MAX_SEQ
    dim = model.get_sentence_embedding_dimension()
    print(f"  Model {model_name} (dim {dim}) on {device}")

    start = 0
    mode = "w+"
    if PROGRESS.exists() and not args.restart:
        p = json.loads(PROGRESS.read_text())
        if p.get("n") == n and p.get("dim") == dim:
            start = p["done"]
            mode = "r+"
            print(f"  Resuming from row {start:,}")

    emb = np.lib.format.open_memmap(OUT_EMB, mode=mode, dtype=np.float32,
                                    shape=(n, dim))
    t0 = time.time()
    for i in range(start, n, args.chunk):
        j = min(i + args.chunk, n)
        v = model.encode(texts[i:j], batch_size=C.EMBED_BATCH,
                         convert_to_numpy=True, show_progress_bar=False,
                         normalize_embeddings=True).astype(np.float32)
        emb[i:j] = v
        emb.flush()
        PROGRESS.write_text(json.dumps({"n": n, "dim": dim, "done": j}))
        el = time.time() - t0
        rate = (j - start) / max(el, 1e-9)
        eta = (n - j) / max(rate, 1e-9) / 60
        print(f"    {j:>9,}/{n:,}  {rate:>6.0f} docs/s  ETA {eta:>5.1f} min",
              flush=True)

    elapsed = time.time() - t0
    slim[[C.ID_COL]].to_csv(OUT_IDS, index=False)
    OUT_MAN.write_text(json.dumps({
        "stage": "2b", "timestamp": datetime.now().isoformat(),
        "model": model_name, "dim": int(dim), "n_records": int(n),
        "normalized": True, "max_seq_length": C.EMBED_MAX_SEQ,
        "device": device, "runtime_min": elapsed / 60,
        "docs_per_sec": (n - start) / max(elapsed, 1e-9),
    }, indent=2))
    if PROGRESS.exists():
        PROGRESS.unlink()
    print(f"\n  Done in {elapsed/60:.1f} min -> {OUT_EMB}")
    print("=" * 72)


if __name__ == "__main__":
    main()
