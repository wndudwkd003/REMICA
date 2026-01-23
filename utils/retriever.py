# utils/retriever.py

import json
from pathlib import Path
from typing import Dict, List, Tuple, Union  # 추가

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

        # key: (ds_name, label) -> (index, meta_sids, meta_texts, meta_labels)
        self.cache: Dict[
            Tuple[str, int],
            Tuple[faiss.Index, List[str], List[str], List[int]],
        ] = {}

        # key: ds_name -> {sid -> label}
        self.sid_to_label_cache: Dict[str, Dict[str, int]] = {}

    # ----------------------------------------------------
    # 1) datasets_processed/*/*.jsonl 에서 sid -> label 매핑 로딩
    # ----------------------------------------------------
    def _load_sid_to_label(self, ds_name: str) -> Dict[str, int]:
        """
        datasets_processed/{ds_name}/{split}.jsonl 들을 읽어서
        id(=sid) -> label 매핑을 한 번 만들어 캐시에 저장합니다.
        """
        if ds_name in self.sid_to_label_cache:
            return self.sid_to_label_cache[ds_name]

        base = Path(self.config.datasets_dir) / ds_name
        sid_to_label: Dict[str, int] = {}

        # 필요한 split 만 쓰고 싶으면 여기 리스트를 조정하시면 됩니다.
        for split in ["train", "valid", "test"]:
            p = base / f"{split}.jsonl"
            if not p.exists():
                continue

            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    s = (line or "").strip()
                    if not s:
                        continue
                    item = json.loads(s)
                    sid = item.get("id")
                    lbl = item.get("label")
                    if sid is None or lbl is None:
                        continue
                    sid_to_label[sid] = int(lbl)

        self.sid_to_label_cache[ds_name] = sid_to_label
        return sid_to_label

    # ----------------------------------------------------
    # 2) FAISS 인덱스 + meta + label 로딩
    # ----------------------------------------------------
    def load(self, ds: Union[DatasetEnum, str], label: int):
        # ds 가 Enum 으로 들어오든, 문자열로 들어오든 처리
        if isinstance(ds, DatasetEnum):
            ds_name = ds.name
        else:
            ds_name = str(ds)

        key = (ds_name, int(label))
        if key in self.cache:
            return self.cache[key]

        base = Path(self.config.sim_index_dir) / ds_name
        index_path = base / f"label{label}.faiss"
        meta_path = base / f"label{label}_meta.jsonl"

        index = faiss.read_index(str(index_path))

        meta_sids: List[str] = []
        meta_texts: List[str] = []
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                meta_sids.append(item["sid"])
                meta_texts.append(item["text"])

        # 여기서 sid -> label 매핑을 불러와서 meta_labels 생성
        sid_to_label = self._load_sid_to_label(ds_name)
        meta_labels: List[int] = []
        for sid in meta_sids:
            if sid not in sid_to_label:
                # 매핑이 없으면 에러를 던지거나, None 을 넣고 나중에 필터링해도 됩니다.
                raise KeyError(
                    f"[SimilarTextRetriever] label not found for sid={sid} in ds={ds_name}"
                )
            meta_labels.append(sid_to_label[sid])

        self.cache[key] = (index, meta_sids, meta_texts, meta_labels)
        return index, meta_sids, meta_texts, meta_labels

    # ----------------------------------------------------
    # 3) 유사 텍스트 + 라벨 함께 반환
    # ----------------------------------------------------
    def get_similar_texts(
        self,
        ds: Union[DatasetEnum, str],
        label: int,
        query_sid: str,
        query_text: str,
        top_k: int,
        extra: int = 10,
    ):
        """
        반환 형식을 기존 "텍스트 리스트"에서
        [{"sid": ..., "text": ..., "label": ...}, ...] 리스트로 확장.
        """
        index, meta_sids, meta_texts, meta_labels = self.load(ds, label)

        q = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        k_search = top_k + extra
        scores, ids = index.search(q, k_search)

        out = []
        for j in ids[0]:
            j = int(j)
            sid = meta_sids[j]

            # 자기 자신은 제외
            if sid == query_sid:
                continue

            ex = {
                "sid": sid,
                "text": meta_texts[j],
                "label": meta_labels[j],
            }
            out.append(ex)
            if len(out) >= top_k:
                break
        return out
