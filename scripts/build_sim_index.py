# scripts/build_sim_index.py

import json
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config.config import Config, REM_STEP_1_DATASET, DatasetEnum
from utils.data_utils import JsonlDataset
from utils.seeds_utils import set_seeds


def build_dataset_index(config: Config, dataset_enum: DatasetEnum, split: str):
    dataset_name = dataset_enum.name
    ds_path = Path(config.datasets_dir) / dataset_name / f"{split}.jsonl"
    print(f"[INFO] Building index for {dataset_name} ({split}) from {ds_path}")
    dataset = JsonlDataset(ds_path, meta_to_text=False)

    sids, texts, labels = [], [], []

    for i in range(len(dataset)):
        sid, text, label, _ = dataset[i]
        sids.append(sid)
        texts.append(text)
        labels.append(label)

    model = SentenceTransformer(config.emb_model)
    emb = model.encode(
        texts,
        batch_size=config.emb_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    out_dir = Path(config.sim_index_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 라벨별로 분리해서 인덱싱
    for y in [0, 1]:
        idxs = [i for i, label in enumerate(labels) if label == y]

        vecs = emb[idxs]
        dim = vecs.shape[1]

        index = faiss.IndexFlatIP(dim)
        index.add(vecs)

        # index 파일
        faiss.write_index(index, str(out_dir / f"label{y}.faiss"))

        # meta 파일 (index의 순서와 동일해야 함)
        meta_path = out_dir / f"label{y}_meta.jsonl"
        with meta_path.open("w", encoding="utf-8") as f:
            for i in idxs:
                f.write(
                    json.dumps({"sid": sids[i], "text": texts[i]}, ensure_ascii=False)
                    + "\n"
                )

        print(f"[OK] {dataset_name} label{y}: n={len(idxs)} -> {out_dir}")


def main():
    config = Config()
    set_seeds(config.seed)

    for ds in REM_STEP_1_DATASET:
        build_dataset_index(config, ds, split="train")

    print("[DONE] All dataset indexes built.")


if __name__ == "__main__":
    main()
