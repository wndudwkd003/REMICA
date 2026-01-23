from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore

from config.config import Config


def main():
    config = Config()

    out_index = Path(config.rem2_faiss_index_path)
    out_meta = Path(config.rem2_faiss_meta_path)
    out_index.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(config.remica_db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    rows = conn.execute(
        """
        SELECT id, memory, reliability, is_correct
        FROM rem_stage_2
        """
    ).fetchall()

    keep = []
    for sid, memory, rel, is_correct in rows:
        m = (memory or "").strip()
        r = float(rel) if rel is not None else 0.0
        c = int(is_correct) if is_correct is not None else 0
        if not m:
            continue
        if r < float(config.rem2_min_reliability):
            continue
        if bool(config.rem2_only_correct) and c != 1:
            continue
        keep.append((str(sid), m))

    device = "cuda" if config.emb_device == "cuda" else "cpu"
    model = SentenceTransformer(str(config.emb_model), device=device)

    texts = [m for _, m in keep]
    embs = model.encode(texts, normalize_embeddings=True)
    embs = np.asarray(embs, dtype=np.float32)

    dim = int(embs.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(embs)

    faiss.write_index(index, str(out_index))

    with out_meta.open("w", encoding="utf-8") as f:
        for sid, _ in keep:
            f.write(json.dumps({"sid": sid}, ensure_ascii=False) + "\n")

    conn.close()

    print(f"[REM2 FAISS] wrote: {out_index}")
    print(f"[REM2 FAISS] wrote: {out_meta}")
    print(f"[REM2 FAISS] n={len(keep)} dim={dim}")


if __name__ == "__main__":
    main()
