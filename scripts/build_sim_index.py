# scripts/build_sim_index.py

from __future__ import annotations

import json
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from config.config import REM_STEP_1_DATASET, Config, DatasetEnum
from utils.data_utils import JsonlDataset
from utils.seeds_utils import set_seeds


def build_dataset_index(config: Config, dataset_enum: DatasetEnum, split: str):
    dataset_name = dataset_enum.name
    ds_path = Path(config.datasets_dir) / dataset_name / f"{split}.jsonl"
    print(f"[INFO] Building unified index for {dataset_name} ({split}) from {ds_path}")

    dataset = JsonlDataset(ds_path, meta_to_text=False)

    sids = []
    texts = []
    labels = []

    for i in range(len(dataset)):
        sid, text, label, _ = dataset[i]
        sids.append(sid)
        texts.append(text)
        labels.append(int(label))

    print(f"[INFO] {dataset_name} ({split}) n={len(sids)} samples")

    # 임베딩 생성 (cosine 유사도용으로 정규화)
    model = SentenceTransformer(config.emb_model)
    emb = model.encode(
        texts,
        batch_size=config.emb_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # 내적 = cosine (정규화 했으므로)

    index.add(emb)
    print(f"[INFO] FAISS index added vectors: {index.ntotal}")

    out_dir = Path(config.sim_index_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 통합 인덱스 파일
    index_path = out_dir / "index.faiss"
    faiss.write_index(index, str(index_path))
    print(f"[OK] wrote index: {index_path}")

    # 메타 파일 (sid, text, label)
    meta_path = out_dir / "meta.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for sid, text, y in zip(sids, texts, labels):
            f.write(
                json.dumps(
                    {"sid": sid, "text": text, "label": int(y)},
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"[OK] wrote meta: {meta_path}")


def main():
    config = Config()
    set_seeds(config.seed)

    # REM_STEP_1_DATASET 에 대해 train split 기준으로 인덱스 생성
    for ds in REM_STEP_1_DATASET:
        build_dataset_index(config, ds, split="train")

    print("[DONE] All unified dataset indexes built.")


if __name__ == "__main__":
    main()
