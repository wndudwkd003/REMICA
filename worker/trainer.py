# worker/trainer.py


import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from config.config import Config, DatasetEnum
from core.build_model import build_model
from utils.collate_utils import TextCollator
from utils.cuda_utils import cleanup_gpu, get_device
from utils.data_utils import JsonlDataset
from utils.dir_utils import make_run_dir
from utils.rem2_retriever import Rem2AugDataset, Rem2Retriever
from utils.viz_utils import plot_cross_grid, plot_train_valid_curves


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
    config: Config,
    run_dir: Path,
    dataset_name: str,
    batch_size: int = 0,
    dataset_sum_mode: bool = False,
) -> Path:

    if dataset_sum_mode:
        train_datasets = []
        valid_datasets = []

        for ds, _ in config.dataset_order:
            paths = dataset_paths(config, ds)
            train_datasets.append(
                JsonlDataset(str(paths["train"]), config.meta_to_text)
            )
            valid_datasets.append(
                JsonlDataset(str(paths["valid"]), config.meta_to_text)
            )

        train_base = ConcatDataset(train_datasets)
        valid_base = ConcatDataset(valid_datasets)

        batch_size = config.dataset_sum_batch_size # 합산 모드 배치 사이즈


    else:
        paths = dataset_paths(config, DatasetEnum[dataset_name])
        train_base = JsonlDataset(str(paths["train"]), config.meta_to_text)
        valid_base = JsonlDataset(str(paths["valid"]), config.meta_to_text)

    # --

    rem2 = None
    if config.use_rem2_aug:
        rem2 = Rem2Retriever(config, device=get_device())

        train_ds = Rem2AugDataset(train_base, rem2)
        valid_ds = Rem2AugDataset(valid_base, rem2)
    else:
        train_ds = train_base
        valid_ds = valid_base

    try:
        best_path = train_once(
            config=config,
            run_dir=run_dir,
            save_name=dataset_name,
            train_ds=train_ds,
            valid_ds=valid_ds,
            batch_size=batch_size,
            desc_tag=dataset_name,
        )
        return best_path
    finally:
        if rem2 is not None:
            rem2.close()


def test_cross(
    config: Config,
    ckpt_path: Path,
    dataset_name: str
) -> dict:

    meta_path = ckpt_path.parent / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    hidden_dim = meta["hidden_dim"]

    model = build_model(config, hidden_dim=hidden_dim, num_labels=2)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    device = get_device()
    model.to(device)

    collate = TextCollator(
        model_id=config.model_id,
        max_len=config.max_len,
        trust_remote_code=True,
    )

    loss_fn = nn.CrossEntropyLoss()
    out = {}

    rem2 = None
    if config.use_rem2_aug:
        rem2 = Rem2Retriever(config, device=device)

    ds_pbar = tqdm(config.dataset_order, desc=f"cross-test({dataset_name})")
    try:
        for test_dataset, bs in ds_pbar:

            paths = dataset_paths(config, test_dataset)
            test_base = JsonlDataset(str(paths["test"]), config.meta_to_text)

            if rem2:
                test_ds = Rem2AugDataset(test_base, rem2)
            else:
                test_ds = test_base

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

    finally:
        if rem2 is not None:
            rem2.close()

        cleanup_gpu(model)
        del model

    return out


def train_once(
    config: Config,
    run_dir: Path,
    save_name: str,
    train_ds,
    valid_ds,
    batch_size: int,
    desc_tag: str,
) -> Path:

    device = get_device()

    collate = TextCollator(
        model_id=config.model_id,
        max_len=config.max_len,
        trust_remote_code=True,
    )

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

    save_dir = run_dir / "save" / save_name
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best.pt"
    meta_path = save_dir / "meta.json"

    best_valid_loss = float("inf")
    best_valid_acc = -1.0
    best_epoch = -1
    bad_count = 0
    history = []

    print(f"\n[TRAIN] dataset={desc_tag} train={len(train_ds)} valid={len(valid_ds)}")
    print(
        f"[TRAIN] early_stopping: patience={config.early_stopping_patience}, "
        f"delta={config.early_stopping_delta} (monitor=valid_loss)"
    )

    epoch_pbar = tqdm(range(1, config.num_epochs + 1), desc=f"epochs({desc_tag})")
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
        title=f"Train/Valid Loss ({desc_tag})",
    )
    plot_train_valid_curves(
        history_csv=history_csv,
        out_path=save_dir / "train_valid_acc.png",
        metric="acc",
        title=f"Train/Valid Acc ({desc_tag})",
    )

    print(
        f"[TRAIN] saved -> {best_path} "
        f"(best_epoch={best_epoch}, best_valid_loss={best_valid_loss:.4f}, best_valid_acc={best_valid_acc:.4f})"
    )

    cleanup_gpu(model)
    del model
    return best_path


def train(config: Config):
    run_dir = make_run_dir(config)
    all_results = {}
    print(f"[NEW RUN] run_dir: {run_dir}")

    if config.dataset_sum:  # 데이터 세트 합산 모드
        sum_name = "SUM"
        ckpt_path = train_model(
            config, run_dir, sum_name, dataset_sum_mode=True
        )
        cross = test_cross(config, ckpt_path, sum_name)
        all_results[sum_name] = cross

        cleanup_gpu()

        print(f"\n[DONE-SUM] run_dir={run_dir}")
        return run_dir

    # 기존: 데이터셋별 개별 학습 모드
    outer = tqdm(config.dataset_order, desc="train-datasets")

    for dataset_enum, bs in outer:
        ckpt_path = train_model(config, run_dir, dataset_enum.name, batch_size=bs)
        cross = test_cross(config, ckpt_path, dataset_enum.name)
        all_results[dataset_enum.name] = cross

        cleanup_gpu()

    print(f"\n[DONE] run_dir={run_dir}")
    return run_dir



def test(config: Config):
    run_dir = Path(config.load_run_dir)

    save_root = run_dir / "save"
    assert save_root.exists(), f"save dir not found: {save_root}"

    all_results = {}

    # 합산 학습 모드일 때: SUM 모델 하나만 불러와서 다시 cross-test
    if config.dataset_sum:
        sum_name = "SUM"
        ckpt_path = save_root / sum_name / "best.pt"
        cross = test_cross(config, ckpt_path, sum_name)
        all_results[sum_name] = cross
        cleanup_gpu()

        write_results(run_dir, config, all_results)

        print(f"\n[DONE-SUM] loaded run_dir={run_dir}")
        return run_dir

    # ----- 기존 per-dataset 모드 -----
    outer = tqdm(config.dataset_order, desc="retest-models")
    for dataset_enum, _ in outer:
        ckpt_path = save_root / dataset_enum.name / "best.pt"
        cross = test_cross(config, ckpt_path, dataset_enum.name)
        all_results[dataset_enum.name] = cross
        cleanup_gpu()

    write_results(run_dir, config, all_results)

    print(f"\n[DONE] loaded run_dir={run_dir}")
    return run_dir



def write_results(
    run_dir: Path,
    config: Config,
    all_results: dict
):
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # JSON 저장
    with open(results_dir / "cross_test.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    # CSV 저장
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

    # 기본 순서: config.dataset_order의 name
    order_names = [ds.name for ds, _ in config.dataset_order]

    # SUM 등 config.dataset_order에 없는 train 이름도 자동으로 뒤에 추가
    for tr in all_results.keys():
        if tr not in order_names:
            order_names.append(tr)

    # 플롯
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
