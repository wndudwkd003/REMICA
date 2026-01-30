# scripts/build_faiss_indices.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from config.config import Config, DatasetEnum, TARGET_DATASETS


# ---------------------------------------------------------
# 유틸: JSONL 로드
# ---------------------------------------------------------


def load_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


# ---------------------------------------------------------
# 유틸: 임베딩 모델 로드 / 텍스트 임베딩
# ---------------------------------------------------------


def load_embed_model(config: Config) -> SentenceTransformer:
    """
    SentenceTransformer 로드. config.rag_model, config.rag_device 사용.
    """
    model = SentenceTransformer(config.rag_model, device=config.rag_device)
    return model


def encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
) -> np.ndarray:
    """
    텍스트 리스트를 배치 단위로 임베딩해서 (N, D) numpy array로 반환.
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # IP + cosine 유사도처럼 쓰기 위해 정규화
    )
    return embeddings.astype("float32")


# ---------------------------------------------------------
# split 단위 처리: 임베딩 + (옵션) FAISS 인덱스
# ---------------------------------------------------------


def process_split(
    config: Config,
    dataset: DatasetEnum,
    split: str,
    model: SentenceTransformer,
    build_index: bool = False,
) -> None:
    """
    단일 데이터셋의 하나의 split(train/valid/test)에 대해:
    - {split}.jsonl 로드
    - text 임베딩
    - embeddings_{split}.npy + meta_{split}.jsonl 저장
    - (build_index=True이면) FAISS IndexFlatIP 생성 및 저장
    """
    base_dir = Path(config.datasets_dir) / dataset.name
    data_path = base_dir / f"{split}.jsonl"

    if not data_path.is_file():
        print(f"[{dataset.name}] {split}.jsonl not found, skip.")
        return

    out_dir = Path(config.faiss_dir) / dataset.name
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = out_dir / f"embeddings_{split}.npy"
    meta_path = out_dir / f"meta_{split}.jsonl"
    index_path = out_dir / f"{split}_index.faiss"

    print(f"\n=== Processing {dataset.name} / {split} ===")
    print(f"- Data path:   {data_path}")
    print(f"- Emb path:    {emb_path}")
    print(f"- Meta path:   {meta_path}")
    if build_index:
        print(f"- Index path:  {index_path}")

    # 1) 데이터 로드 (id, text, label)
    ids: List[str] = []
    texts: List[str] = []
    labels: List[int] = []

    for row in tqdm(load_jsonl(data_path), desc=f"Loading {dataset.name} [{split}]"):
        rid = row.get("id")
        text = row.get("text", "")
        label = row.get("label", None)

        if not isinstance(text, str) or not text.strip():
            continue

        if rid is None:
            rid = f"{dataset.name}_{split}_noid_{len(ids)}"

        ids.append(str(rid))
        texts.append(text.strip())
        labels.append(label)

    if not texts:
        print(f"- No valid texts found in {data_path}, skip split.")
        return

    print(f"- Loaded {len(texts)} samples for {dataset.name} [{split}]")

    # 2) 임베딩 계산
    emb = encode_texts(
        model=model,
        texts=texts,
        batch_size=config.rag_batch_size,
    )
    n, dim = emb.shape
    print(f"- Embedding shape: {emb.shape} (N={n}, D={dim})")

    # 3) 임베딩 .npy 저장
    np.save(emb_path, emb)
    print(f"- Saved embeddings to: {emb_path}")

    # 4) 메타데이터 저장
    with meta_path.open("w", encoding="utf-8") as f:
        for rid, text, label in zip(ids, texts, labels):
            rec = {
                "id": rid,
                "text": text,
                "label": label,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"- Saved metadata to: {meta_path}")

    # 5) (옵션) FAISS 인덱스 생성 및 저장
    if build_index:
        index = faiss.IndexFlatIP(dim)
        index.add(emb)
        print(f"- FAISS index size: {index.ntotal}")
        faiss.write_index(index, str(index_path))
        print(f"- Saved FAISS index to: {index_path}")


# ---------------------------------------------------------
# 메인: TARGET_DATASETS × {train,valid,test}
# ---------------------------------------------------------


def main() -> None:
    config = Config()

    print("Config settings:")
    print(f"- datasets_dir:   {config.datasets_dir}")
    print(f"- faiss_dir:      {config.faiss_dir}")
    print(f"- rag_model:      {config.rag_model}")
    print(f"- rag_device:     {config.rag_device}")
    print(f"- rag_batch_size: {config.rag_batch_size}")
    print(f"- TARGET_DATASETS: {[d.name for d in TARGET_DATASETS]}")

    # 임베딩 모델 미리 한 번만 로드해서 재사용
    model = load_embed_model(config)

    splits = ["train", "valid", "test"]

    for ds in TARGET_DATASETS:
        for split in splits:
            # train 에 대해서만 FAISS 인덱스 생성, valid/test는 임베딩+메타만
            build_index = split == "train"
            process_split(config, ds, split, model, build_index=build_index)

    print("\nAll embeddings (and train indices) built successfully.")


if __name__ == "__main__":
    main()
