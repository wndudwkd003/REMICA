# scripts/build_ica_faiss.py
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from utils.db_utils import open_db
from params.db_value import DB


def build_ica_faiss(
    db_path: str,
    out_dir: str,
    emb_model: str,
    emb_device: str = "cpu",
    batch_size: int = 128,
    split: str | None = None,
    index_field: str = "last_text",  # "last_text" or "mix"
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) DB load
    conn = open_db(db_path)
    table = DB.ICA.value

    # 필요한 컬럼만 가져오기
    cols = [
        DB.ID.value,
        DB.CONVERSATION_LABEL.value,
        DB.SOURCE_DATASET.value,
        DB.SPLIT.value,
        DB.SOURCE_CID.value,
        DB.WINDOW_START.value,
        DB.WINDOW_END.value,
        DB.LAST_TEXT.value,
        DB.CONTEXT_SUMMARY.value,
        DB.RULES_JSON.value,
    ]

    q = f"SELECT {', '.join(cols)} FROM {table}"
    args = ()
    if split is not None:
        q += f" WHERE {DB.SPLIT.value}=?"
        args = (split,)

    rows = conn.execute(q, args).fetchall()
    conn.close()

    if not rows:
        raise RuntimeError("ICA rows are empty. Check DB path / split filter.")

    # 2) build texts + meta
    texts: list[str] = []
    meta: list[dict] = []

    for r in rows:
        sid = str(r[0])
        clabel = int(r[1])
        source_dataset = str(r[2])
        split_val = str(r[3])
        source_cid = str(r[4])
        w0 = int(r[5])
        w1 = int(r[6])
        last_text = str(r[7] or "").strip()
        summary = str(r[8] or "").strip()
        rules_json = str(r[9] or "[]")

        if index_field == "mix":
            # 필요시 혼합 버전(실험용)
            try:
                rules = json.loads(rules_json)
                if not isinstance(rules, list):
                    rules = []
                rules = [str(x).strip() for x in rules if str(x).strip()]
            except Exception:
                rules = []
            mix = last_text
            if summary:
                mix += "\n" + summary
            if rules:
                mix += "\nRules:\n" + "\n".join(rules)
            text_for_index = mix.strip()
        else:
            text_for_index = last_text

        # 빈 텍스트는 제외(인덱스 품질 저하)
        if not text_for_index:
            continue

        texts.append(text_for_index)
        meta.append(
            {
                "sid": sid,
                "conversation_label": clabel,
                "source_dataset": source_dataset,
                "split": split_val,
                "source_cid": source_cid,
                "window_start": w0,
                "window_end": w1,
            }
        )

    if not texts:
        raise RuntimeError("All ICA texts are empty after filtering.")

    # 3) embed
    model = SentenceTransformer(emb_model, device=emb_device)

    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine용 핵심
    ).astype(np.float32)

    d = int(embs.shape[1])

    # 4) faiss index (cosine = inner product with normalized vectors)
    index = faiss.IndexFlatIP(d)
    index.add(embs)

    # 5) save
    tag = f"{index_field}" + (f"_{split}" if split else "")
    index_path = out_dir / f"ica_{tag}.index"
    meta_path = out_dir / f"ica_{tag}.meta.jsonl"

    faiss.write_index(index, str(index_path))

    with meta_path.open("w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"[OK] index saved: {index_path}")
    print(f"[OK] meta  saved: {meta_path}")
    print(f"[OK] vectors: {index.ntotal} dim={d}")


if __name__ == "__main__":
    # 필요에 맞게 경로/모델명만 바꿔서 실행
    build_ica_faiss(
        db_path="rem/remica.sqlite3",
        out_dir="rem/faiss",
        emb_model="sentence-transformers/all-MiniLM-L6-v2",
        emb_device="cuda",
        batch_size=128,
        split="train",
        index_field="last_text",
    )
