
# REMICA: Reflective Memory and Interventional Context Alignment with Multi-Agent LLMs for Inappropriate Utterance Detection

REMICA mitigates OOD generalization issues caused by cross-dataset norm shifts and reduces the latency/cost and error lock-in risk of online multi-agent debate. It builds offline reflective memories of agents’ judgments and normative rationales (REM), reuses them via similarity retrieval for single-pass inference, and aligns rationale–prediction with label intervention (ICA).

## Repo Content (GitHub)
This repository contains code only:

- `client/`, `config/`, `enums/`, `utils/`, `worker/`
- `scripts/`, `latex_scripts/`
- `run.py`

Large artifacts are excluded:

- `datasets/`, `datasets_processed/`, `debate_memory/`, `runs/`, `faiss_index/`

## Memory
Large artifacts (e.g., debate memory) are uploaded to Google Drive:
https://drive.google.com/drive/folders/1TbNgqTwjc0RzKNBQbxi-JVDHZbTUsZsW?usp=sharing

> **Privacy note**
> - The original text field (`text`) has been removed from the Drive artifacts for release.
> - Each sample is referenced only by an identifier in the format `<DATASET>_<split>_<idx>` (e.g., `ToxiSpanSE_train_41124`).


## Data Processing & Indexing
- Preprocessing: `scripts/data_collection.py`
- FAISS index build: `scripts/build_faiss_indices.py`

## Datasets
Experiments use 4 English datasets:
- `DiaSafety`
- `ToxiSpanSE`
- `CHSD` (referred to as `HSDCD` in the code)
- `RTP` (referred to as `RealToxicityPrompts` in the code)

The codebase is implemented to support additional datasets (see `DatasetEnum`) and will be extended in future work.

### Processed Data Format
Processed samples are stored in JSONL with the following schema:

```json
{
  "id": "<DATASET>_<split>_<idx>",
  "text": "<here is text>",
  "label": 0,
  "metadata": {
    "context": "<here is context>",
    "category": "<here is category>"
  }
}
````

All text fields (e.g., `text`, `metadata.context`) are anonymized in the released artifacts.



## Config

Main config: `config/config.py`

* `do_mode`: `SAVE_DEBATE_MEMORY` / `TEST_DEBATE_MEMORY` / `ONLINE_TEST`
* `chain_mode`: `chain_of_expert` / `chain_of_debate` / `none`
* paths: `memory_dir=debate_memory`, `datasets_dir=datasets_processed`, `faiss_dir=faiss_index`
* API config: `config/api.json` (may contain private keys → do not commit)

## Run

```bash
python -m run
```
