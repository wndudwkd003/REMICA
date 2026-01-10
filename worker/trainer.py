# worker/trainer.py


import gc
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from config.config import Config, DatasetEnum
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from utils.data_utils import JsonlDataset
from utils.collate_utils import TextCollator
from core.build_model import build_model
from utils.viz_utils import plot_cross_grid
from utils.viz_utils import plot_train_valid_curves


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_gpu(model: nn.Module | None = None):
    try:
        if model is not None:
            model.to("cpu")
    except Exception:
        pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_run_dir(config: Config) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{ts}_{config.model_name}"
    run_dir = Path(config.run_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "save").mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)

    cfg = asdict(config) if is_dataclass(config) else dict(config.__dict__)
    cfg["dataset_order"] = [(ds.name, bs) for ds, bs in config.dataset_order]

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    return run_dir


def dataset_paths(config: Config, dataset: DatasetEnum):
    base = Path(config.datasets_dir) / dataset.name
    return {
        "train": base / "train.jsonl",
        "valid": base / "valid.jsonl",
        "test": base / "test.jsonl",
    }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    *,
    train: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    model_id: str = "",
    desc: str = "",
):
    if train:
        model.train()
    else:
        model.eval()

    total = 0
    correct = 0
    sum_loss = 0.0

    pbar = tqdm(loader, desc=desc, leave=False)

    with torch.set_grad_enabled(train):
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            if "longformer" in (model_id or "").lower():
                # Longformer: global_attention_mask 필요/권장
                global_attention_mask = torch.zeros_like(attn)
                global_attention_mask[:, 0] = 1
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attn,
                    global_attention_mask=global_attention_mask,
                )
            else:
                logits = model(input_ids=input_ids, attention_mask=attn)

            loss = loss_fn(logits, y)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()

            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()

            bs = y.size(0)
            total += bs
            sum_loss += loss.item() * bs

            cur_loss = sum_loss / max(total, 1)
            cur_acc = correct / max(total, 1)
            pbar.set_postfix(loss=f"{cur_loss:.4f}", acc=f"{cur_acc:.4f}")

    return {
        "loss": sum_loss / max(total, 1),
        "acc": correct / max(total, 1),
        "n": float(total),
    }


def train_model(
    config: Config, run_dir: Path, train_dataset: "DatasetEnum", batch_size: int
) -> Path:
    device = get_device()
    paths = dataset_paths(config, train_dataset)

    collate = TextCollator(
        model_id=config.model_id,
        max_len=config.max_len,
        trust_remote_code=True,
    )

    train_ds = JsonlDataset(str(paths["train"]), meta_to_text=config.meta_to_text)
    valid_ds = JsonlDataset(str(paths["valid"]), meta_to_text=config.meta_to_text)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collate,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate,
    )

    model = build_model(config, hidden_dim=config.hidden_dim, num_labels=2).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    total_steps = config.num_epochs * len(train_loader)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    loss_fn = nn.CrossEntropyLoss()

    save_dir = run_dir / "save" / train_dataset.name
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best.pt"
    meta_path = save_dir / "meta.json"

    best_valid_loss = float("inf")
    best_valid_acc = -1.0
    best_epoch = -1
    bad_count = 0
    history = []

    print(
        f"\n[TRAIN] dataset={train_dataset.name} train={len(train_ds)} valid={len(valid_ds)}"
    )
    print(
        f"[TRAIN] early_stopping: patience={config.early_stopping_patience}, "
        f"delta={config.early_stopping_delta} (monitor=valid_loss)"
    )

    epoch_pbar = tqdm(
        range(1, config.num_epochs + 1), desc=f"epochs({train_dataset.name})"
    )
    for epoch in epoch_pbar:
        tr = run_epoch(
            model,
            train_loader,
            device,
            loss_fn,
            train=True,
            optimizer=optimizer,
            scheduler=scheduler,
            model_id=config.model_id,
            desc=f"train e{epoch:02d}",
        )
        va = run_epoch(
            model,
            valid_loader,
            device,
            loss_fn,
            train=False,
            model_id=config.model_id,
            desc=f"valid e{epoch:02d}",
        )

        history.append(
            {
                "epoch": epoch,
                "tr_loss": tr["loss"],
                "tr_acc": tr["acc"],
                "va_loss": va["loss"],
                "va_acc": va["acc"],
            }
        )

        epoch_pbar.set_postfix(
            tr_loss=f"{tr['loss']:.4f}",
            tr_acc=f"{tr['acc']:.4f}",
            va_loss=f"{va['loss']:.4f}",
            va_acc=f"{va['acc']:.4f}",
        )

        improved = (best_valid_loss - va["loss"]) > config.early_stopping_delta

        if improved:
            best_valid_loss = va["loss"]
            best_valid_acc = va["acc"]
            best_epoch = epoch
            bad_count = 0

            torch.save(model.state_dict(), best_path)

            meta = {
                "model_id": config.model_id,
                "hidden_dim": config.hidden_dim,
                "max_len": config.max_len,
                "meta_to_text": config.meta_to_text,
                "best_epoch": best_epoch,
                "best_valid_loss": best_valid_loss,
                "best_valid_acc": best_valid_acc,
                "batch_size": batch_size,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        else:
            bad_count += 1
            if bad_count >= config.early_stopping_patience:
                break

    history_csv = save_dir / "train_valid_history.csv"
    pd.DataFrame(history).to_csv(history_csv, index=False)

    plot_train_valid_curves(
        history_csv=history_csv,
        out_path=save_dir / "train_valid_loss.png",
        metric="loss",
        title=f"Train/Valid Loss ({train_dataset.name})",
    )
    plot_train_valid_curves(
        history_csv=history_csv,
        out_path=save_dir / "train_valid_acc.png",
        metric="acc",
        title=f"Train/Valid Acc ({train_dataset.name})",
    )

    print(
        f"[TRAIN] saved -> {best_path} "
        f"(best_epoch={best_epoch}, best_valid_loss={best_valid_loss:.4f}, best_valid_acc={best_valid_acc:.4f})"
    )

    cleanup_gpu(model)
    del model
    return best_path


def test_cross(config: Config, ckpt_path: Path, train_dataset: "DatasetEnum"):
    device = get_device()

    meta_path = ckpt_path.parent / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    hidden_dim = meta["hidden_dim"]

    model = build_model(config, hidden_dim=hidden_dim, num_labels=2)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    collate = TextCollator(
        model_id=config.model_id,
        max_len=config.max_len,
        trust_remote_code=True,
    )

    loss_fn = nn.CrossEntropyLoss()
    out = {}

    ds_pbar = tqdm(config.dataset_order, desc=f"cross-test({train_dataset.name})")
    for test_dataset, bs in ds_pbar:
        paths = dataset_paths(config, test_dataset)
        test_ds = JsonlDataset(str(paths["test"]), meta_to_text=config.meta_to_text)
        test_loader = DataLoader(
            test_ds,
            batch_size=bs,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate,
        )

        m = run_epoch(
            model,
            test_loader,
            device,
            loss_fn,
            train=False,
            model_id=config.model_id,
            desc=f"test {test_dataset.name}",
        )

        out[test_dataset.name] = m
        ds_pbar.set_postfix(acc=f"{m['acc']:.4f}", loss=f"{m['loss']:.4f}")

    cleanup_gpu(model)
    del model
    return out


def train(config: Config):
    run_dir = make_run_dir(config)
    all_results = {}

    outer = tqdm(config.dataset_order, desc="train-datasets")
    for train_dataset, bs in outer:
        ckpt_path = train_model(config, run_dir, train_dataset, batch_size=bs)
        cross = test_cross(config, ckpt_path, train_dataset)
        all_results[train_dataset.name] = cross

        results_dir = run_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)

        with open(results_dir / "cross_test.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        rows = []
        for tr, d in all_results.items():
            for te, m in d.items():
                rows.append(
                    {
                        "train_dataset": tr,
                        "test_dataset": te,
                        "acc": m["acc"],
                        "loss": m["loss"],
                        "n": m["n"],
                    }
                )

        csv_path = results_dir / "cross_test.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)

        order_names = [ds.name for ds, _ in config.dataset_order]

        plot_cross_grid(
            csv_path=csv_path,
            out_path=results_dir / "cross_test_grid_acc.png",
            metric="acc",
            dataset_order=order_names,
            title="Cross-test Accuracy",
            fmt=".3f",
        )
        plot_cross_grid(
            csv_path=csv_path,
            out_path=results_dir / "cross_test_grid_loss.png",
            metric="loss",
            dataset_order=order_names,
            title="Cross-test Loss",
            fmt=".3f",
        )

        cleanup_gpu()

    print(f"\n[DONE] run_dir={run_dir}")
    return run_dir


def test(config: Config):
    run_dir = Path(config.load_run_dir)

    save_root = run_dir / "save"
    assert save_root.exists(), f"save dir not found: {save_root}"

    all_results = {}

    outer = tqdm(config.dataset_order, desc="retest-models")
    for dataset, _ in outer:
        ckpt_path = save_root / dataset.name / "best.pt"
        cross = test_cross(config, ckpt_path, dataset)
        all_results[dataset.name] = cross
        cleanup_gpu()

    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "cross_test_retest.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    rows = []
    for tr, d in all_results.items():
        for te, m in d.items():
            rows.append(
                {
                    "train_dataset": tr,
                    "test_dataset": te,
                    "acc": m["acc"],
                    "loss": m["loss"],
                    "n": m["n"],
                }
            )

    csv_path = results_dir / "cross_test_retest.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    order_names = [ds.name for ds, _ in config.dataset_order]

    plot_cross_grid(
        csv_path=csv_path,
        out_path=results_dir / "cross_test_retest_grid_acc.png",
        metric="acc",
        dataset_order=order_names,
        title="Cross-test Accuracy (Retest)",
        fmt=".3f",
    )
    plot_cross_grid(
        csv_path=csv_path,
        out_path=results_dir / "cross_test_retest_grid_loss.png",
        metric="loss",
        dataset_order=order_names,
        title="Cross-test Loss (Retest)",
        fmt=".3f",
    )

    print(f"\n[DONE] loaded run_dir={run_dir}")
    return run_dir
