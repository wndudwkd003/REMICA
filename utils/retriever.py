# utils/retriever.py

import json
from pathlib import Path
from typing import Union

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config.config import Config, DatasetEnum


class SimilarTextRetriever:
    def __init__(
        self,
        config: Config,
        device: str,
    ):
        self.config = config
        self.model_id = config.emb_model
        self.device = device

        self.model = SentenceTransformer(self.model_id, device=self.device)

        # key: ds_name -> (index, meta_sids, meta_texts, meta_labels)
        self.cache = {}

    # ----------------------------------------------------
    # 1) 통합 인덱스 로드
    # ----------------------------------------------------
    def load(self, ds: Union[DatasetEnum, str]):
        if isinstance(ds, DatasetEnum):
            ds_name = ds.name
        else:
            ds_name = str(ds)

        if ds_name in self.cache:
            return self.cache[ds_name]

        base = Path(self.config.sim_index_dir) / ds_name
        index_path = base / "index.faiss"
        meta_path = base / "meta.jsonl"

        index = faiss.read_index(str(index_path))

        meta_sids = []
        meta_texts = []
        meta_labels = []

        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                meta_sids.append(item["sid"])
                meta_texts.append(item["text"])
                meta_labels.append(int(item["label"]))

        self.cache[ds_name] = (index, meta_sids, meta_texts, meta_labels)
        return self.cache[ds_name]

    # ----------------------------------------------------
    # 2) 유사 텍스트 검색 (라벨 필터 없음)
    # ----------------------------------------------------
    def get_similar_texts(
        self,
        ds: Union[DatasetEnum, str],
        query_sid: str,
        query_text: str,
        top_k: int,
        extra: int = 10,
    ):
        index, meta_sids, meta_texts, meta_labels = self.load(ds)

        q = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        k_search = top_k + extra
        _, ids = index.search(q, k_search)

        out = []
        seen = set()

        for j in ids[0]:
            j = int(j)
            if j < 0 or j >= len(meta_sids):
                continue

            sid = meta_sids[j]
            if sid == query_sid:
                continue
            if sid in seen:
                continue
            seen.add(sid)

            out.append(
                {
                    "sid": sid,
                    "text": meta_texts[j],
                    "label": int(meta_labels[j]),
                }
            )
            if len(out) >= top_k:
                break

        return out
