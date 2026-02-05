# utils/faiss_utils.py

from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config.config import Config, DatasetEnum
from utils.data_utils import load_jsonl


_EMBED_MODEL_CACHE = {}


def get_embed_model(config: Config) -> SentenceTransformer:
    key = (config.rag_model, config.rag_device)
    model = _EMBED_MODEL_CACHE.get(key)
    if model is not None:
        return model

    model = SentenceTransformer(config.rag_model, device=config.rag_device)
    _EMBED_MODEL_CACHE[key] = model
    return model


def encode_query(config: Config, text: str) -> np.ndarray:
    model = get_embed_model(config)
    emb = model.encode(
        [text],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb[0].astype("float32")


def load_faiss_index_and_meta(
    config: Config, dataset: DatasetEnum, split: str = "train"
):
    base_dir = Path(config.faiss_dir) / dataset.name
    index_path = base_dir / f"{split}_index.faiss"
    meta_path = base_dir / f"meta_{split}.jsonl"

    if not meta_path.is_file():
        raise FileNotFoundError(f"meta file not found: {meta_path}")
    if not index_path.is_file():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")

    meta = list(load_jsonl(meta_path))
    if not meta:
        raise RuntimeError(f"No records in meta file: {meta_path}")

    index = faiss.read_index(str(index_path))
    if index.ntotal != len(meta):
        raise RuntimeError(
            f"Index size ({index.ntotal}) and meta length ({len(meta)}) mismatch "
            f"for {dataset.name} [{split}]"
        )

    return index, meta


def search_similar_by_text(
    config: Config,
    dataset: DatasetEnum,
    index: faiss.Index,
    meta_train,
    query_text: str,
    top_k: int,
    query_id: str,
):
    if top_k <= 0:
        return []

    if index.ntotal == 0:
        raise RuntimeError(f"Empty FAISS index for dataset={dataset.name}")

    q = encode_query(config, query_text).reshape(1, -1)

    k = min(top_k + 1, index.ntotal)
    scores, idxs = index.search(q, k)

    scores = scores[0]
    idxs = idxs[0]

    results = []
    for score, idx in zip(scores, idxs):
        if idx < 0 or idx >= len(meta_train):
            continue

        rec = meta_train[idx]

        if str(rec["id"]) == str(query_id):
            continue

        results.append(
            {
                "id": str(rec["id"]),
                "text": rec["text"],
                "label": rec["label"],
                "score": float(score),
            }
        )

        if len(results) >= top_k:
            break

    return results


def pick_topk_with_both_labels_in_order(
    candidates_full: list[dict],
    use_k: int,
):

    topk = candidates_full[:use_k]

    if use_k == 1:
        return topk


    labels = {int(x["label"]) for x in topk}

    # 이미 0/1 섞여 있으면 그대로
    if len(labels) >= 2:
        return topk

    # top-k가 단일 라벨이면 반대 라벨이 후보 전체에 있는지 확인
    only_label = next(iter(labels))
    other_label = 1 - only_label

    first_other_idx = None
    for i in range(use_k, len(candidates_full)):
        if int(candidates_full[i]["label"]) == other_label:
            first_other_idx = i
            break

    if first_other_idx is None:
        raise ValueError(
            f"cannot ensure both labels within candidates: only label={only_label}"
        )

    # 유사도 순 유지: topk에서 마지막 1개 제거 + 반대라벨의 가장 이른 후보 1개 추가
    # 결과 순서는 topk[:-1] (원래 앞쪽) 다음에 (더 뒤에 있는 other 후보)로 유지됨
    out = topk[:-1] + [candidates_full[first_other_idx]]

    return out
