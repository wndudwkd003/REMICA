# utils/process_utils.py

from __future__ import annotations

import torch

from config.config import Config, DatasetEnum
from enums.out_schema import AgentAOut, AgentBOut, GPTInferOut
from utils.faiss_utils import load_faiss_index_and_meta
from utils.llm_builder import build_agent_client

G_GPU_ID = None
G_CONFIG = None
G_CLIENTS = None
G_INDEX_CACHE = None


def init_worker(gpu_id: int, config: Config) -> None:
    global G_GPU_ID, G_CONFIG, G_CLIENTS, G_INDEX_CACHE

    G_GPU_ID = int(gpu_id)
    G_CONFIG = config

    torch.cuda.set_device(G_GPU_ID)
    G_CONFIG.rag_device = f"cuda:{G_GPU_ID}"

    G_CLIENTS = {
        "A": build_agent_client(G_CONFIG, AgentAOut, role_key="A"),
        "B": build_agent_client(G_CONFIG, AgentBOut, role_key="B"),
        "INFER": None,
    }

    G_INDEX_CACHE = {}


def get_index_meta(dataset: DatasetEnum):
    key = dataset.name
    if key in G_INDEX_CACHE:
        return G_INDEX_CACHE[key]

    index, meta_train = load_faiss_index_and_meta(G_CONFIG, dataset, "train")
    G_INDEX_CACHE[key] = (index, meta_train)
    return index, meta_train


def ensure_infer_client():
    if G_CLIENTS["INFER"] is None:
        G_CLIENTS["INFER"] = build_agent_client(G_CONFIG, GPTInferOut, role_key="INFER")
    return G_CLIENTS["INFER"]


def validate_step_output(step_name: str, out_raw: dict) -> dict:
    if step_name.startswith("A"):
        return AgentAOut.model_validate(out_raw).model_dump()
    if step_name.startswith("B"):
        return AgentBOut.model_validate(out_raw).model_dump()
    raise ValueError(f"Unknown step_name: {step_name}")
