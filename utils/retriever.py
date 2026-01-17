# utils/retriever.py

from config.config import Config, DatasetEnum
from pathlib import Path

from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np


class SimilarTextRetriever:
    def __init__(self, config: Config):
        self.config = config
        self.model_id = config.emb_model

        device = config.emb_device
        self.model = SentenceTransformer(self.model_id, device=device)

        self.cache = {}

    def load(self, ds: DatasetEnum, label: int):
        key = (ds.name, int(label))
        if key in self.cache:
            return self.cache[key]

        base = Path(self.config.sim_index_dir) / ds.name
        index_path = base / f"label{int(label)}.faiss"
        meta_path = base / f"label{int(label)}_meta.jsonl"

        index = faiss.read_index(str(index_path))

        meta_sids = []
        meta_texts = []
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                meta_sids.append(item["sid"])
                meta_texts.append(item["text"])

        self.cache[key] = (index, meta_sids, meta_texts)
        return index, meta_sids, meta_texts

    def get_similar_texts(
        self, ds, label, query_sid, query_text, top_k, extra: int = 16
    ):
        index, meta_sids, meta_texts = self.load(ds, label)

        q = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        k_search = min(len(meta_texts), top_k + extra)
        scores, ids = index.search(q, k_search)

        out = []
        for j in ids[0]:
            sid = meta_sids[int(j)]
            if sid == query_sid:
                continue
            out.append(meta_texts[int(j)])
            if len(out) >= top_k:
                break
        return out
