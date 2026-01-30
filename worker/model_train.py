# worker/model_train.py

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from anthropic import Anthropic
from openai import OpenAI
from tqdm.auto import tqdm

from config.config import Config, DatasetEnum
from utils.data_utils import load_jsonl
from utils.metrics_utils import compute_binary_metrics, save_metrics_csv_and_plots
from utils.time_utils import now_ms


# ---------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------


def _is_openai_model(model_name: str) -> bool:
    return model_name.startswith("gpt-")


def _is_claude_model(model_name: str) -> bool:
    return model_name.startswith("claude-")


# ---------------------------------------------------------
# Prompt builders (shared) - fine-tune version: NO dataset perspective
# ---------------------------------------------------------


def _build_system_prompt(dataset: DatasetEnum) -> str:
    return (
        "You are Agent a binary appropriateness classifier for hate/toxicity detection.\n\n"
        "[Output format]\n"
        "Return a single JSON object that matches this schema:\n"
        "{\n"
        '  "pred_label": 0 or 1,\n'
        '  "rationale_structured": "string"\n'
        "}\n\n"
        "[Constraints]\n"
        "pred_label MUST be 0 (appropriate) or 1 (inappropriate).\n"
        "All free-text fields (including rationale_structured) MUST NOT contain any quote characters (no \" and no ').\n"
        "Do NOT output any extra text before or after the JSON object.\n"
    )


def _build_user_prompt(text: str) -> str:
    return "Classify the following text.\n\n" "Text:\n" f"{text}\n"


def _label_to_rationale(label: int) -> str:
    if label == 0:
        return "appropriate"
    return "inappropriate"


def _build_assistant_json(label: int) -> str:
    obj = {
        "pred_label": int(label),
        "rationale_structured": _label_to_rationale(int(label)),
    }
    return json.dumps(obj, ensure_ascii=False)


def _openai_response_format_json_schema() -> dict:
    # Chat Completions response_format:
    # { "type": "json_schema", "json_schema": { "name": "...", "schema": {...}, "strict": true } }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "binary_appropriateness",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["pred_label", "rationale_structured"],
                "properties": {
                    "pred_label": {"type": "integer", "enum": [0, 1]},
                    "rationale_structured": {"type": "string"},
                },
            },
        },
    }


# ---------------------------------------------------------
# Training record builders (provider-specific wrappers)
# ---------------------------------------------------------


def _build_openai_training_record(dataset: DatasetEnum, text: str, label: int) -> dict:
    system_prompt = _build_system_prompt(dataset)
    user_content = _build_user_prompt(text)
    assistant_content = _build_assistant_json(int(label))

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def _build_claude_training_record(dataset: DatasetEnum, text: str, label: int) -> dict:
    system_prompt = _build_system_prompt(dataset)
    user_content = _build_user_prompt(text)
    assistant_content = _build_assistant_json(int(label))

    return {
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    }


# ---------------------------------------------------------
# Build unified fine-tuning JSONL from datasets_processed/*
# ---------------------------------------------------------


def build_finetune_jsonl(config: Config) -> Path:
    model_name = config.api_model.value

    out_dir = Path(config.run_dir) / "finetune"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train_messages.jsonl"

    n_samples = 0

    with out_path.open("w", encoding="utf-8") as f:
        for dataset, _bs in config.dataset_order:
            dataset_dir = Path(config.datasets_dir) / dataset.name
            train_path = dataset_dir / "train.jsonl"
            if not train_path.is_file():
                print(
                    f"[build_finetune_jsonl] train.jsonl not found, skip: {train_path}"
                )
                continue

            print(f"[build_finetune_jsonl] dataset={dataset.name} | {train_path}")

            for row in load_jsonl(train_path):
                text = str(row["text"]).strip()
                label_int = int(row["label"])

                if label_int not in (0, 1):
                    continue
                if not text:
                    continue

                if _is_openai_model(model_name):
                    rec = _build_openai_training_record(dataset, text, label_int)
                elif _is_claude_model(model_name):
                    rec = _build_claude_training_record(dataset, text, label_int)
                else:
                    raise ValueError(
                        f"[build_finetune_jsonl] Unsupported model for fine-tune: {model_name}"
                    )

                f.write(json.dumps(rec, ensure_ascii=False))
                f.write("\n")
                n_samples += 1

    print(f"[build_finetune_jsonl] wrote {n_samples} samples → {out_path}")
    return out_path


# ---------------------------------------------------------
# Claude fine-tuning helpers
# ---------------------------------------------------------


def _upload_claude_training_file(client: Anthropic, file_path: Path) -> str:
    print(f"[Claude] Uploading training file: {file_path}")

    with file_path.open("rb") as f:
        file_obj = client.files.create(file=f, purpose="fine-tune")

    print(f"[Claude] File uploaded: {file_obj.id}")
    return file_obj.id


def _create_claude_finetune_job(
    client: Anthropic,
    base_model: str,
    training_file_id: str,
    n_epochs: int,
) -> dict:
    print(f"[Claude] Creating fine-tune job for base model: {base_model}")

    job = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model=base_model,
        hyperparameters={"n_epochs": n_epochs},
        suffix="remica-binary-classifier",
    )

    return {
        "job_id": job.id,
        "base_model": base_model,
        "training_file": training_file_id,
        "status": job.status,
        "created_at": getattr(job, "created_at", None),
    }


# ---------------------------------------------------------
# MODEL_TRAIN: create fine-tune job (OpenAI or Claude)
# ---------------------------------------------------------


def run_model_train(config: Config) -> None:
    model_name = config.api_model.value
    train_file_path = build_finetune_jsonl(config)

    if _is_openai_model(model_name):
        client = OpenAI()

        print(f"[MODEL_TRAIN] Uploading training file: {train_file_path}")
        with train_file_path.open("rb") as f:
            file_obj = client.files.create(file=f, purpose="fine-tune")

        print(f"[MODEL_TRAIN] Creating fine-tune job for base model: {model_name}")
        job = client.fine_tuning.jobs.create(
            training_file=file_obj.id,
            model=model_name,
            hyperparameters={"n_epochs": config.num_epochs},
            suffix="remica-binary-classifier",
        )

        meta = {
            "job_id": job.id,
            "base_model": model_name,
            "training_file": file_obj.id,
            "status": job.status,
        }

        out_dir = train_file_path.parent
        meta_path = out_dir / "openai_finetune_job.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print("[MODEL_TRAIN] OpenAI fine-tune job created:")
        print(f"  - job_id: {job.id}")
        print(f"  - status: {job.status}")
        print(f"  - meta saved: {meta_path}")
        return

    if _is_claude_model(model_name):
        client = Anthropic()

        file_id = _upload_claude_training_file(client, train_file_path)

        meta = _create_claude_finetune_job(
            client=client,
            base_model=model_name,
            training_file_id=file_id,
            n_epochs=config.num_epochs,
        )

        out_dir = train_file_path.parent
        meta_path = out_dir / "claude_finetune_job.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print("[MODEL_TRAIN] Claude fine-tune job created:")
        print(f"  - job_id: {meta['job_id']}")
        print(f"  - status: {meta['status']}")
        print(f"  - meta saved: {meta_path}")
        return

    raise ValueError(f"[MODEL_TRAIN] Unsupported model backend: {model_name}")


# ---------------------------------------------------------
# Prediction helpers (provider-specific)
# ---------------------------------------------------------


def _parse_pred_label_from_json(content: str) -> int:
    obj = json.loads(content)
    pred_label = int(obj["pred_label"])
    if pred_label not in (0, 1):
        raise ValueError(f"pred_label out of range: {pred_label}")
    return pred_label


def _predict_label_openai(
    client: OpenAI,
    model_name: str,
    dataset: DatasetEnum,
    text: str,
    config: Config,
) -> int:
    system_prompt = _build_system_prompt(dataset)
    user_content = _build_user_prompt(text)

    resp = client.chat.completions.create(
        model=model_name,
        response_format=_openai_response_format_json_schema(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=config.max_new_tokens["DIRECT"],
        temperature=config.temperature,
        top_p=config.top_p,
    )

    content = resp.choices[0].message.content.strip()
    return _parse_pred_label_from_json(content)


def _predict_label_claude(
    client: Anthropic,
    model_name: str,
    dataset: DatasetEnum,
    text: str,
    config: Config,
) -> int:
    system_prompt = _build_system_prompt(dataset)
    user_content = _build_user_prompt(text)

    resp = client.messages.create(
        model=model_name,
        max_tokens=config.max_new_tokens["DIRECT"],
        temperature=config.temperature,
        top_p=config.top_p,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )

    content = ""
    for block in resp.content:
        if getattr(block, "type", None) == "text":
            content += block.text

    content = content.strip()
    return _parse_pred_label_from_json(content)


# ---------------------------------------------------------
# MODEL_TEST: evaluate fine-tuned model on test.jsonl
# ---------------------------------------------------------


def run_model_test(config: Config) -> None:
    model_name = config.api_model.value
    finetune_dir = Path(config.run_dir) / "finetune"

    if _is_openai_model(model_name):
        meta_path = finetune_dir / "openai_finetune_job.json"
        if not meta_path.is_file():
            print(f"[MODEL_TEST] fine-tune meta not found: {meta_path}")
            return

        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        job_id = meta["job_id"]

        client = OpenAI()
        job = client.fine_tuning.jobs.retrieve(job_id)

        print(f"[MODEL_TEST] job_id={job.id}, status={job.status}")

        if job.status != "succeeded":
            print(
                "[MODEL_TEST] Fine-tuning job has not succeeded yet. Abort evaluation."
            )
            return

        ft_model = job.fine_tuned_model
        if not ft_model:
            raise ValueError(
                "[MODEL_TEST] fine_tuned_model is empty even though status=succeeded."
            )

        print(f"[MODEL_TEST] Using fine-tuned model: {ft_model}")

        _evaluate_model_on_datasets(
            config=config,
            predict_fn=lambda dataset, text: _predict_label_openai(
                client, ft_model, dataset, text, config
            ),
            model_identifier=ft_model,
            mode="model_test_finetune",
        )
        return

    if _is_claude_model(model_name):
        meta_path = finetune_dir / "claude_finetune_job.json"
        if not meta_path.is_file():
            print(f"[MODEL_TEST] fine-tune meta not found: {meta_path}")
            return

        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        job_id = meta["job_id"]

        client = Anthropic()
        job = client.fine_tuning.jobs.retrieve(job_id)

        print(f"[MODEL_TEST] job_id={job.id}, status={job.status}")

        if job.status != "succeeded":
            print(
                "[MODEL_TEST] Fine-tuning job has not succeeded yet. Abort evaluation."
            )
            return

        ft_model = job.fine_tuned_model
        if not ft_model:
            raise ValueError(
                "[MODEL_TEST] fine_tuned_model is empty even though status=succeeded."
            )

        print(f"[MODEL_TEST] Using fine-tuned model: {ft_model}")

        _evaluate_model_on_datasets(
            config=config,
            predict_fn=lambda dataset, text: _predict_label_claude(
                client, ft_model, dataset, text, config
            ),
            model_identifier=ft_model,
            mode="model_test_finetune",
        )
        return

    raise ValueError(f"[MODEL_TEST] Unsupported model backend: {model_name}")


def _evaluate_model_on_datasets(
    config: Config,
    predict_fn,
    model_identifier: str,
    mode: str,
) -> None:
    for dataset, _bs in config.dataset_order:
        dataset_dir = Path(config.datasets_dir) / dataset.name
        test_path = dataset_dir / "test.jsonl"
        if not test_path.is_file():
            print(f"[MODEL_TEST] test.jsonl not found, skip: {test_path}")
            continue

        print(f"[MODEL_TEST] evaluating dataset={dataset.name} | {test_path}")

        y_true = []
        y_pred = []
        times_ms = []

        for row in tqdm(load_jsonl(test_path), desc=f"ft-test {dataset.name}[test]"):
            text = str(row["text"]).strip()
            gold = int(row["label"])

            if gold not in (0, 1):
                continue
            if not text:
                continue

            y_true.append(gold)

            t0 = now_ms()
            pred = predict_fn(dataset, text)
            t1 = now_ms()

            y_pred.append(pred)
            times_ms.append(t1 - t0)

        if not y_true:
            print(f"[MODEL_TEST] No valid samples for dataset={dataset.name}")
            continue

        metrics = compute_binary_metrics(y_true, y_pred)
        save_metrics_csv_and_plots(
            config=config,
            dataset=dataset,
            mode=mode,
            metrics=metrics,
        )

        time_info = {
            "n_samples": len(times_ms),
            "median_ms": float(np.median(times_ms)),
            "mean_ms": float(np.mean(times_ms)),
        }
        out_time = (
            Path(config.run_dir) / mode / f"{dataset.name}_{model_identifier}_time.json"
        )
        out_time.parent.mkdir(parents=True, exist_ok=True)
        with out_time.open("w", encoding="utf-8") as f:
            json.dump(time_info, f, ensure_ascii=False, indent=2)

        print(f"[MODEL_TEST] metrics & time saved for {dataset.name}")
