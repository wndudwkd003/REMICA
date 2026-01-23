# utils/rem2_retriever.py

import json
import sqlite3
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config.config import Config
from utils.data_utils import JsonlDataset


class Rem2Retriever:
    def __init__(self, config: Config, device: str | torch.device):
        self.db_path = config.remica_db_path # REMICA DB 경로
        self.top_k = config.rem2_top_k # 유사 사례 최대 개수
        self.min_rel = config.rem2_min_reliability # 최소 신뢰도
        self.only_correct = config.rem2_only_correct # 정답인 사례만 사용 여부

        self.index_path = Path(config.rem2_faiss_index_path)
        self.meta_path = Path(config.rem2_faiss_meta_path)

        self.model = SentenceTransformer(str(config.emb_model), device=device)
        self.index = faiss.read_index(str(self.index_path))

        self.meta = []
        with self.meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.meta.append(json.loads(line))

        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")

    def embed(self, text: str) -> np.ndarray:
        v = self.model.encode([str(text)], normalize_embeddings=True)
        return np.asarray(v, dtype=np.float32)

    def fetch_stage2_rows(self, sids: list[str]) -> list[dict]:
        if not sids:
            return []
        q = ",".join(["?"] * len(sids))
        sql = f"""
        SELECT id, evidence, memory, reliability, is_correct
        FROM rem_stage_2
        WHERE id IN ({q})
        """
        rows = self.conn.execute(sql, sids).fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "sid": r[0],
                    "evidence": r[1] or "",
                    "memory": r[2] or "",
                    "reliability": float(r[3]) if r[3] is not None else 0.0,
                    "is_correct": int(r[4]) if r[4] is not None else 0,
                }
            )
        return out

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def search_memories(
        self, query_text: str, exclude_sid: str | None = None
    ) -> list[str]:
        if self.top_k <= 0:
            return []

        q = self.embed(query_text)
        k = max(self.top_k * 8, self.top_k)
        _, I = self.index.search(q, k)

        cand_sids = []
        for idx in I[0].tolist():
            if idx < 0 or idx >= len(self.meta):
                continue
            sid = self.meta[idx].get("sid")
            if not sid:
                continue
            if exclude_sid and sid == exclude_sid:
                continue
            cand_sids.append(sid)

        if not cand_sids:
            return []

        rows = self.fetch_stage2_rows(cand_sids)

        rows = [r for r in rows if r["reliability"] >= self.min_rel]
        if self.only_correct:
            rows = [r for r in rows if r["is_correct"] == 1]

        rows.sort(key=lambda x: x["reliability"], reverse=True)

        mems = []
        for r in rows:
            m = (r["memory"] or "").strip()
            if not m:
                continue
            mems.append(m)
            if len(mems) >= self.top_k:
                break

        return mems


class Rem2AugDataset:
    def __init__(self, base: JsonlDataset, retriever: Rem2Retriever):
        self.base = base
        self.retriever = retriever

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        sid, text, label, meta = self.base[idx]
        mems = self.retriever.search_memories(text, exclude_sid=sid)
        if mems:
            mem_block = "\n".join([f"- {m}" for m in mems])
            text = f"{text}\n\nREM2_MEMORIES:\n{mem_block}"
        return sid, text, label, meta
