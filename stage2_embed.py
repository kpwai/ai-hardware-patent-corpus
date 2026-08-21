#!/usr/bin/env python3
"""
STAGE 2 -- Embeddings, keyed by ID.

The v1 failure mode was aligning a CSV to an .npy by row position after a read
that silently skipped malformed lines. Three rows went missing and the pipeline
truncated to hide it. This stage makes that impossible: every embedding row is
written alongside its Lens ID, and Stage 3 joins on the ID.

Outputs:
    stage2_embeddings/embeddings.npy      float32, unit-normalised
    stage2_embeddings/embedding_ids.csv   one Lens ID per row, same order
    stage2_embeddings/manifest.json

Run:  python stage2_embed.py
"""

import json
import time
from datetime import datetime

import numpy as np
import pandas as pd

import config as C


def main():
    print("=" * 72)
    print("STAGE 2 -- EMBEDDINGS")
    print("=" * 72)

    corpus = pd.read_parquet(C.STAGE1 / "corpus.parquet")
    n = len(corpus)
    print(f"  Corpus: {n:,} records")

    texts = corpus["text"].fillna("").astype(str).tolist()
    ids = corpus[C.ID_COL].astype(str).tolist()

    empty = sum(1 for t in texts if not t.strip())
    if empty:
        print(f"  WARNING: {empty} records have empty text. They are KEPT and")
        print("           embedded as empty strings, so row counts stay aligned.")
        print("           Filtering them here is what broke v1.")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit("pip install sentence-transformers")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"    {torch.cuda.get_device_name(0)}")

    model_name = C.EMBED_MODEL
    try:
        model = SentenceTransformer(model_name, device=device)
    except Exception as e:
        print(f"  Could not load {model_name} ({e}); falling back to "
              f"{C.EMBED_FALLBACK}")
        model_name = C.EMBED_FALLBACK
        model = SentenceTransformer(model_name, device=device)

    model.max_seq_length = C.EMBED_MAX_SEQ
    dim = model.get_sentence_embedding_dimension()
    print(f"  Model: {model_name}  (dim {dim}, max_seq {C.EMBED_MAX_SEQ})")

    t0 = time.time()
    emb = model.encode(
        texts,
        batch_size=C.EMBED_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=C.EMBED_NORMALIZE,
    ).astype(np.float32)
    elapsed = time.time() - t0

    assert emb.shape[0] == n, f"FATAL: {emb.shape[0]} embeddings vs {n} records"

    norms = np.linalg.norm(emb, axis=1)
    print(f"\n  Shape {emb.shape}  | norms mean {norms.mean():.4f} "
          f"std {norms.std():.4f}")
    print(f"  Runtime {elapsed/60:.1f} min ({n/max(elapsed,1):.0f} docs/sec)")

    np.save(C.STAGE2 / "embeddings.npy", emb)
    pd.DataFrame({C.ID_COL: ids}).to_csv(C.STAGE2 / "embedding_ids.csv",
                                         index=False)

    manifest = {
        "stage": 2,
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "dim": int(dim),
        "n_records": n,
        "normalized": C.EMBED_NORMALIZE,
        "max_seq_length": C.EMBED_MAX_SEQ,
        "batch_size": C.EMBED_BATCH,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "runtime_min": elapsed / 60,
        "docs_per_sec": n / max(elapsed, 1),
        "empty_texts": empty,
    }
    (C.STAGE2 / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n  Wrote -> {C.STAGE2}")
    print("  manifest.json holds the model/runtime details for the paper.")
    print("=" * 72)


if __name__ == "__main__":
    main()
