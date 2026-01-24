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
from utils.collate_utils import LMCollator, TextCollator
from utils.cuda_utils import cleanup_gpu, get_device
from utils.data_utils import JsonlDataset
from utils.dir_utils import make_run_dir
from utils.llm_eval import eval_llm_classifier
from utils.rem2_lm_dataset import Rem2LMDataset
from utils.rem2_retriever import Rem2ExampleAugDataset, Rem2Retriever
from utils.viz_utils import plot_cross_grid, plot_train_valid_curves


# ---------------------------------------------------------
# 0. 공통: 데이터 경로
# ---------------------------------------------------------
def dataset_paths(config: Config, dataset: DatasetEnum):
    base = Path(config.datasets_dir) / dataset.name
    return {
        "train": base / "train.jsonl",
        "valid": base / "valid.jsonl",
        "test": base / "test.jsonl",
    }


# ---------------------------------------------------------
# 1. 공통: 한 epoch (CLS / LLM 둘 다 여기서 처리)
# ---------------------------------------------------------
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    train: bool,
    optimizer=None,
    scheduler=None,
    loss_fn: nn.Module | None = None,
    model_id: str = "",
    desc: str = "",
    is_llm: bool = False,
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

            if is_llm:
                # LLM: CausalLM loss
                labels = batch["labels"].to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attn,
                    labels=labels,
                )
                loss = outputs.loss
            else:
                # CLS: encoder + MLP
                assert loss_fn is not None, "CLS 모드에서는 loss_fn이 필요합니다."
                labels = batch["labels"].to(device)

                # longformer 전용 global attention
                if "longformer" in (model_id or "").lower():
                    global_attention_mask = torch.zeros_like(attn)
                    global_attention_mask[:, 0] = 1
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attn,
                        global_attention_mask=global_attention_mask,
                    )
                else:
                    logits = model(input_ids=input_ids, attention_mask=attn)

                loss = loss_fn(logits, labels)
                pred = logits.argmax(dim=-1)
                correct += (pred == labels).sum().item()

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            bs = input_ids.size(0)
            total += bs
            sum_loss += loss.item() * bs

            cur_loss = sum_loss / max(total, 1)
            if is_llm:
                pbar.set_postfix(loss=f"{cur_loss:.4f}")
            else:
                cur_acc = correct / max(total, 1)
                pbar.set_postfix(loss=f"{cur_loss:.4f}", acc=f"{cur_acc:.4f}")

    out = {
        "loss": sum_loss / max(total, 1),
        "n": float(total),
    }
    if not is_llm:
        out["acc"] = correct / max(total, 1)

    return out


# ---------------------------------------------------------
# 2. 공통: train/valid JsonlDataset 레지스트리
#   - 여기서는 단순히 (DatasetEnum, JsonlDataset) 리스트만 만든다.
#   - 배치사이즈는 여기서 만지지 않는다.
# ---------------------------------------------------------
def build_base_pairs(config: Config, dataset_name: str, dataset_sum_mode: bool):
    if dataset_sum_mode:
        train_pairs: list[tuple[DatasetEnum, JsonlDataset]] = []
        valid_pairs: list[tuple[DatasetEnum, JsonlDataset]] = []
        for ds, _ in config.dataset_order:
            paths = dataset_paths(config, ds)
            train_pairs.append(
                (ds, JsonlDataset(str(paths["train"]), config.meta_to_text))
            )
            valid_pairs.append(
                (ds, JsonlDataset(str(paths["valid"]), config.meta_to_text))
            )
    else:
        ds_enum = DatasetEnum[dataset_name]
        paths = dataset_paths(config, ds_enum)
        train_pairs = [(ds_enum, JsonlDataset(str(paths["train"]), config.meta_to_text))]
        valid_pairs = [(ds_enum, JsonlDataset(str(paths["valid"]), config.meta_to_text))]

    return train_pairs, valid_pairs


def train_once(
    config: Config,
    run_dir: Path,
    save_name: str,
    train_ds,
    valid_ds,
    batch_size: int,
    desc_tag: str,
    use_llm: bool,
) -> Path:
    device = get_device()

    # -------- 모델 / collator 준비 --------
    model = build_model(config, hidden_dim=config.hidden_dim, num_labels=2, llm=use_llm)
    model_id = config.model_id
    max_len = config.max_len

    if use_llm:
        collate = LMCollator(
            model_id=config.model_id,
            max_len=max_len,
            trust_remote_code=True,
        )
        tokenizer = collate.tokenizer
        loss_fn = None
    else:
        tokenizer = None  # CLS에서는 사용하지 않음
        collate = TextCollator(
            model_id=config.model_id,
            max_len=config.max_len,
            trust_remote_code=True,
        )
        loss_fn = nn.CrossEntropyLoss()

    model.to(device)

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

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    total_steps = config.num_epochs * len(train_loader)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # -------- 저장 경로 / early stopping 상태 --------
    save_dir = run_dir / "save" / save_name
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best.pt"
    meta_path = save_dir / "meta.json"

    best_valid_loss = float("inf")
    best_valid_acc = -1.0  # CLS용
    best_epoch = -1
    bad_count = 0
    history: list[dict] = []

    tag = "LLM" if use_llm else "CLS"
    print(
        f"\n[TRAIN-{tag}] dataset={desc_tag} "
        f"train={len(train_ds)} valid={len(valid_ds)}"
    )
    print(
        f"[TRAIN-{tag}] early_stopping: patience={config.early_stopping_patience}, "
        f"delta={config.early_stopping_delta} (monitor=valid_loss)"
    )

    epoch_desc = f"epochs-llm({desc_tag})" if use_llm else f"epochs({desc_tag})"
    epoch_pbar = tqdm(range(1, config.num_epochs + 1), desc=epoch_desc)

    for epoch in epoch_pbar:
        tr = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            train=True,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            model_id=model_id,
            desc=f"train{'-llm' if use_llm else ''} e{epoch:02d}",
            is_llm=use_llm,
        )
        va = run_epoch(
            model=model,
            loader=valid_loader,
            device=device,
            train=False,
            optimizer=None,
            scheduler=None,
            loss_fn=loss_fn,
            model_id=model_id,
            desc=f"valid{'-llm' if use_llm else ''} e{epoch:02d}",
            is_llm=use_llm,
        )

        if use_llm:
            history.append(
                {
                    "epoch": epoch,
                    "tr_loss": tr["loss"],
                    "va_loss": va["loss"],
                }
            )
            epoch_pbar.set_postfix(
                tr_loss=f"{tr['loss']:.4f}",
                va_loss=f"{va['loss']:.4f}",
            )
        else:
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
            best_epoch = epoch
            bad_count = 0

            torch.save(model.state_dict(), best_path)

            meta: dict = {
                "model_id": config.model_id,
                "max_len": max_len,
                "best_epoch": best_epoch,
                "best_valid_loss": best_valid_loss,
                "batch_size": batch_size,
            }
            if use_llm:
                meta["use_llm_classifier"] = True
            else:
                meta["hidden_dim"] = config.hidden_dim
                meta["meta_to_text"] = config.meta_to_text
                meta["best_valid_acc"] = va["acc"]
                best_valid_acc = va["acc"]

            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        else:
            bad_count += 1
            if bad_count >= config.early_stopping_patience:
                break

    # -------- 히스토리 저장 / 플롯 --------
    if use_llm:
        history_csv = save_dir / "train_valid_history_llm.csv"
        pd.DataFrame(history).to_csv(history_csv, index=False)

        plot_train_valid_curves(
            history_csv=history_csv,
            out_path=save_dir / "train_valid_loss_llm.png",
            metric="loss",
            title=f"Train/Valid Loss LLM ({desc_tag})",
        )
        print(
            f"[TRAIN-LLM] saved -> {best_path} "
            f"(best_epoch={best_epoch}, best_valid_loss={best_valid_loss:.4f})"
        )
    else:
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
            f"[TRAIN-CLS] saved -> {best_path} "
            f"(best_epoch={best_epoch}, best_valid_loss={best_valid_loss:.4f}, "
            f"best_valid_acc={best_valid_acc:.4f})"
        )

    # -------- LLM일 때는 여기서 <answer> 파싱 평가 --------
    if use_llm:
        eval_dir = save_dir / "llm_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)

        metrics = eval_llm_classifier(
            model=model,
            tokenizer=tokenizer,
            dataset=valid_ds,
            device=device,
            max_len=max_len,
            batch_size=batch_size,
            gen_max_new_tokens=config.max_len,
            out_dir=eval_dir,
            desc=f"eval-llm({desc_tag})",
        )

        print(
            f"[EVAL-LLM] acc={metrics['acc']:.4f}, "
            f"f1={metrics['f1']:.4f}, "
            f"parsed_ratio={metrics['parsed_ratio']:.3f} "
            f"(n_total={metrics['n_total']}, n_parsed={metrics['n_parsed']})"
        )

        # meta.json 업데이트
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_loaded = json.load(f)
        except Exception:
            meta_loaded = {
                "model_id": config.model_id,
                "max_len": max_len,
                "use_llm_classifier": True,
                "best_epoch": best_epoch,
                "best_valid_loss": best_valid_loss,
                "batch_size": batch_size,
            }

        meta_loaded["llm_eval_metrics"] = metrics
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_loaded, f, ensure_ascii=False, indent=2)

    cleanup_gpu(model)
    del model
    return best_path


def train_model(
    config: Config,
    run_dir: Path,
    dataset_name: str,
    batch_size: int | None = None,
    dataset_sum_mode: bool = False,
) -> Path:
    use_llm = config.use_llm_classifier

    train_pairs, valid_pairs = build_base_pairs(
        config, dataset_name, dataset_sum_mode
    )


    bs = config.dataset_sum_batch_size if dataset_sum_mode else batch_size

    rem_retriever = None
    try:
        if dataset_sum_mode:
            if config.use_rem2_aug:
                # SUM + REM2 증강
                rem_retriever = Rem2Retriever(config, device=get_device())
                train_base = Rem2ExampleAugDataset(
                    datasets=train_pairs,
                    retriever=rem_retriever,
                    top_k=config.rem2_top_k,
                    mode=config.rem_mode,
                )
                valid_base = Rem2ExampleAugDataset(
                    datasets=valid_pairs,
                    retriever=rem_retriever,
                    top_k=config.rem2_top_k,
                    mode=config.rem_mode,
                )
            else:
                # SUM + 순수 Concat
                train_base = ConcatDataset([ds for _, ds in train_pairs])
                valid_base = ConcatDataset([ds for _, ds in valid_pairs])
        else:
            # ===== 개별 데이터셋 모드 =====
            if config.use_rem2_aug:
                # 단일 데이터셋 + REM2 증강
                rem_retriever = Rem2Retriever(config, device=get_device())
                train_base = Rem2ExampleAugDataset(
                    datasets=[train_pairs[0]],
                    retriever=rem_retriever,
                    top_k=config.rem2_top_k,
                    mode=config.rem_mode,
                )
                valid_base = Rem2ExampleAugDataset(
                    datasets=[valid_pairs[0]],
                    retriever=rem_retriever,
                    top_k=config.rem2_top_k,
                    mode=config.rem_mode,
                )
            else:
                # 단일 데이터셋 + 순수 JsonlDataset
                train_base = train_pairs[0][1]
                valid_base = valid_pairs[0][1]

        # 4) LLM / CLS에 따라 최종 데이터셋 결정
        if use_llm:
            train_ds = Rem2LMDataset(base_dataset=train_base, config=config)
            valid_ds = Rem2LMDataset(base_dataset=valid_base, config=config)
        else:
            train_ds = train_base
            valid_ds = valid_base

        ckpt_path = train_once(
            config=config,
            run_dir=run_dir,
            save_name=dataset_name,
            train_ds=train_ds,
            valid_ds=valid_ds,
            batch_size=bs,
            desc_tag=dataset_name,
            use_llm=use_llm,
        )
        return ckpt_path
    finally:
        if rem_retriever is not None:
            rem_retriever.close()

# ---------------------------------------------------------
# 5. CLS cross-test (LLM에는 적용 안 함)
# ---------------------------------------------------------
def test_cross(
    config: Config,
    ckpt_path: Path,
    dataset_name: str,
) -> dict:
    meta_path = ckpt_path.parent / "meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    hidden_dim = meta["hidden_dim"]

    model = build_model(config, hidden_dim=hidden_dim, num_labels=2, llm=False)
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
    out: dict = {}

    rem_retriever = Rem2Retriever(config, device=device) if config.use_rem2_aug else None

    ds_pbar = tqdm(config.dataset_order, desc=f"cross-test({dataset_name})")
    try:
        for test_dataset, bs in ds_pbar:
            paths = dataset_paths(config, test_dataset)
            test_base = JsonlDataset(str(paths["test"]), config.meta_to_text)

            if rem_retriever is not None:
                test_pairs = [(test_dataset, test_base)]
                test_ds = Rem2ExampleAugDataset(
                    datasets=test_pairs,
                    retriever=rem_retriever,
                    top_k=config.rem2_top_k,
                    mode=config.rem_mode,
                )
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
                model=model,
                loader=test_loader,
                device=device,
                train=False,
                optimizer=None,
                scheduler=None,
                loss_fn=loss_fn,
                model_id=config.model_id,
                desc=f"test {test_dataset.name}",
                is_llm=False,
            )

            out[test_dataset.name] = m
            ds_pbar.set_postfix(acc=f"{m['acc']:.4f}", loss=f"{m['loss']:.4f}")
    finally:
        if rem_retriever is not None:
            rem_retriever.close()
        cleanup_gpu(model)
        del model

    return out


# ---------------------------------------------------------
# 6. train / test / write_results
# ---------------------------------------------------------
def train(config: Config):
    run_dir = make_run_dir(config)
    all_results: dict = {}
    print(f"[NEW RUN] run_dir: {run_dir}")

    use_llm = config.use_llm_classifier

    # SUM 모드
    if config.dataset_sum:
        sum_name = "SUM"
        ckpt_path = train_model(
            config=config,
            run_dir=run_dir,
            dataset_name=sum_name,
            dataset_sum_mode=True,
        )
        if not use_llm:
            cross = test_cross(config, ckpt_path, sum_name)
            all_results[sum_name] = cross
            cleanup_gpu()
            write_results(run_dir, config, all_results)
            print(f"\n[DONE-SUM] run_dir={run_dir}")
        else:
            print(f"\n[DONE-SUM-LLM] run_dir={run_dir}")
        return run_dir

    # 개별 데이터셋 모드
    outer = tqdm(config.dataset_order, desc="train-datasets")
    for dataset_enum, bs in outer:
        ckpt_path = train_model(
            config=config,
            run_dir=run_dir,
            dataset_name=dataset_enum.name,
            batch_size=bs,
        )
        if not use_llm:
            cross = test_cross(config, ckpt_path, dataset_enum.name)
            all_results[dataset_enum.name] = cross
            cleanup_gpu()

    if not use_llm:
        write_results(run_dir, config, all_results)
        print(f"\n[DONE] run_dir={run_dir}")
    else:
        print(f"\n[DONE-LLM] run_dir={run_dir}")
    return run_dir


def test(config: Config):
    run_dir = Path(config.load_run_dir)
    save_root = run_dir / "save"
    assert save_root.exists(), f"save dir not found: {save_root}"

    use_llm = config.use_llm_classifier

    if use_llm:
        print("[TEST] LLM classifier에 대한 cross-test는 trainer에서 따로 구현하지 않았습니다.")
        return run_dir

    all_results: dict = {}

    if config.dataset_sum:
        sum_name = "SUM"
        ckpt_path = save_root / sum_name / "best.pt"
        cross = test_cross(config, ckpt_path, sum_name)
        all_results[sum_name] = cross
        cleanup_gpu()
        write_results(run_dir, config, all_results)
        print(f"\n[DONE-SUM] loaded run_dir={run_dir}")
        return run_dir

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
    all_results: dict,
):
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
    for tr in all_results.keys():
        if tr not in order_names:
            order_names.append(tr)

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
