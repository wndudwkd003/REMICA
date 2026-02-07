# scripts/analysis_rag_match.py

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression

from config.config import Config, DatasetEnum, TARGET_DATASETS
from utils.seeds_utils import set_seeds
from utils.faiss_utils import load_faiss_index_and_meta, search_similar_by_text


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_jsonl_id_text_label(jsonl_path: Path):
    ids = []
    texts = []
    labels = []

    for obj in iter_jsonl(jsonl_path):
        ids.append(str(obj["id"]))
        texts.append(str(obj["text"]))
        labels.append(int(obj["label"]))

    return ids, texts, labels


def embed_texts(model: SentenceTransformer, texts: list[str], batch_size: int):
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32)


def make_2d_tsne(
    vectors: np.ndarray,
    seed: int,
    perplexity: int,
    n_iter: int,
    pca_dim: int,
):
    x = vectors
    if pca_dim > 0 and x.shape[1] > pca_dim:
        x = PCA(n_components=pca_dim, random_state=seed).fit_transform(x)

    tsne = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=perplexity,
        max_iter=n_iter,
        init="pca",
        learning_rate="auto",
        metric="cosine",
    )
    z = tsne.fit_transform(x)
    return z.astype(np.float32)


def dataset_display_name(dataset: DatasetEnum) -> str:
    name = dataset.name

    if name.upper() == "HSDCD":
        return "CHSD"

    up = name.upper()
    if "REAL" in up and ("TOXICITY" in up or "TOXITIY" in up) and "PROMPT" in up:
        return "RTP"

    return name


def save_points_csv(
    out_path: Path,
    points: np.ndarray,
    split: list[str],
    label: list[int],
    ids: list[str],
):
    with out_path.open("w", encoding="utf-8") as f:
        f.write("x,y,split,label,id\n")
        for (x, y), sp, lb, sid in zip(points, split, label, ids):
            f.write(f"{float(x)},{float(y)},{sp},{int(lb)},{sid}\n")


def plot_scatter(
    out_path: Path,
    points: np.ndarray,
    split: list[str],
    label: list[int],
    title: str,
):
    style_map = {
        ("train", 0): {"marker": "o", "color": "#d81b60"},  # deep pink
        ("train", 1): {"marker": "o", "color": "#1e88e5"},  # deep blue
        ("test", 0): {"marker": "^", "color": "#6a1b9a"},   # deep purple
        ("test", 1): {"marker": "^", "color": "#2e7d32"},   # deep green
    }

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111)

    combos = [
        ("train", 0),
        ("train", 1),
        ("test", 0),
        ("test", 1),
    ]

    split_arr = np.array(split)
    label_arr = np.array(label, dtype=int)

    for sp, lb in combos:
        mask = (split_arr == sp) & (label_arr == lb)
        if not np.any(mask):
            continue

        style = style_map[(sp, lb)]
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=18,
            marker=style["marker"],
            c=style["color"],
            alpha=0.85,
            linewidths=0.0,
            label=f"{sp} / label={lb}",
        )

    ax.set_title(title)
    ax.set_xlabel("t-SNE dim-1")
    ax.set_ylabel("t-SNE dim-2")
    ax.legend(loc="best", frameon=True)
    ax.grid(True, linewidth=0.3, alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def compute_match_stats(per_sample: list[dict], k: int):
    sums = {0: 0.0, 1: 0.0}
    counts = {0: 0, 1: 0}

    for r in per_sample:
        y = int(r["true_label"])
        v = float(r["match_frac"])
        sums[y] += v
        counts[y] += 1

    mean0 = float(sums[0] / counts[0]) if counts[0] > 0 else 0.0
    mean1 = float(sums[1] / counts[1]) if counts[1] > 0 else 0.0
    overall = (
        float((sums[0] + sums[1]) / (counts[0] + counts[1]))
        if (counts[0] + counts[1]) > 0
        else 0.0
    )

    out = {
        "k": int(k),
        "metric_name": f"label_purity@{int(k)}",
        "metric_definition": "mean over test samples of (# neighbors with same label as test) / K",
        "by_true_label": {
            "0": {"n": int(counts[0]), "mean_label_purity": mean0},
            "1": {"n": int(counts[1]), "mean_label_purity": mean1},
        },
        "n_total": int(counts[0] + counts[1]),
        "overall_mean_label_purity": overall,
    }

    return out


def save_per_sample_csv(out_path: Path, per_sample: list[dict]):
    with out_path.open("w", encoding="utf-8") as f:
        f.write("id,true_label,k,match_count,match_frac,retrieved_labels\n")
        for r in per_sample:
            sid = str(r["id"])
            y = int(r["true_label"])
            k = int(r["k"])
            mc = int(r["match_count"])
            mf = float(r["match_frac"])
            labs = " ".join([str(int(x)) for x in r["retrieved_labels"]])
            f.write(f"{sid},{y},{k},{mc},{mf:.6f},{labs}\n")


def plot_match_bar(out_path: Path, dataset_disp: str, k: int, stats: dict):
    y0 = float(stats["by_true_label"]["0"]["mean_label_purity"])
    y1 = float(stats["by_true_label"]["1"]["mean_label_purity"])

    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.add_subplot(111)

    ax.bar(["true=0", "true=1"], [y0, y1])

    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(f"mean label_purity@{k}")
    ax.set_title(f"{dataset_disp} | mean label_purity@{k} by true label")
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def compute_signed_distance_to_boundary(train_emb: np.ndarray, train_labels: list[int], emb: np.ndarray, seed: int):
    y = np.array(train_labels, dtype=int)

    clf = LogisticRegression(
        random_state=seed,
        max_iter=2000,
        solver="lbfgs",
    )
    clf.fit(train_emb, y)

    w = clf.coef_.reshape(-1)          # (d,)
    b = float(clf.intercept_.reshape(-1)[0])

    w_norm = float(np.linalg.norm(w))
    if w_norm == 0.0:
        signed = np.zeros((emb.shape[0],), dtype=np.float32)
        absd = np.zeros((emb.shape[0],), dtype=np.float32)
        return signed, absd

    signed = (emb @ w + b) / w_norm
    absd = np.abs(signed)

    return signed.astype(np.float32), absd.astype(np.float32)


def boundary_summary(abs_dist: np.ndarray, labels: list[int], thresholds: list[float]):
    y = np.array(labels, dtype=int)

    out = {
        "n_total": int(abs_dist.shape[0]),
        "thresholds_abs_distance": [float(t) for t in thresholds],
        "overall": {},
        "by_label": {},
    }

    for t in thresholds:
        frac = float(np.mean(abs_dist <= t)) if abs_dist.size > 0 else 0.0
        out["overall"][str(float(t))] = {"fraction_near_boundary": frac}

    for lb in [0, 1]:
        mask = y == lb
        dist_lb = abs_dist[mask]
        if dist_lb.size == 0:
            out["by_label"][str(lb)] = {
                "n": 0,
                "mean_abs_distance": 0.0,
                "median_abs_distance": 0.0,
                "near_boundary": {},
            }
            continue

        near = {}
        for t in thresholds:
            frac = float(np.mean(dist_lb <= t))
            near[str(float(t))] = {"fraction_near_boundary": frac}

        out["by_label"][str(lb)] = {
            "n": int(dist_lb.size),
            "mean_abs_distance": float(np.mean(dist_lb)),
            "median_abs_distance": float(np.median(dist_lb)),
            "near_boundary": near,
        }

    return out


def save_boundary_per_sample_csv(
    out_path: Path,
    ids: list[str],
    labels: list[int],
    signed_dist: np.ndarray,
    abs_dist: np.ndarray,
):
    with out_path.open("w", encoding="utf-8") as f:
        f.write("id,label,signed_distance,abs_distance\n")
        for sid, y, sd, ad in zip(ids, labels, signed_dist, abs_dist):
            f.write(f"{sid},{int(y)},{float(sd):.8f},{float(ad):.8f}\n")


def plot_abs_distance_hist(out_path: Path, dataset_disp: str, split_name: str, abs_dist: np.ndarray, labels: list[int]):
    y = np.array(labels, dtype=int)

    d0 = abs_dist[y == 0]
    d1 = abs_dist[y == 1]

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    # matplotlib 기본 색 사용(사용자님 선호: 색 지정 최소화)
    ax.hist(d0, bins=50, alpha=0.65, label="label=0")
    ax.hist(d1, bins=50, alpha=0.65, label="label=1")

    ax.set_title(f"{dataset_disp} | {split_name} | abs(distance to boundary)")
    ax.set_xlabel("abs(distance)")
    ax.set_ylabel("count")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run_one_dataset(config: Config, dataset: DatasetEnum):
    dataset_dir = Path(config.datasets_dir) / dataset.name
    train_path = dataset_dir / "train.jsonl"
    test_path = dataset_dir / "test.jsonl"

    train_ids, train_texts, train_labels = load_jsonl_id_text_label(train_path)
    test_ids, test_texts, test_labels = load_jsonl_id_text_label(test_path)

    index, meta_train = load_faiss_index_and_meta(config, dataset, split="train")

    k = int(config.rag_top_k)
    if k <= 0:
        raise ValueError("config.rag_top_k must be > 0 for match@K analysis")

    per_sample = []
    for sid, text, y in zip(test_ids, test_texts, test_labels):
        retrieved = search_similar_by_text(
            config=config,
            dataset=dataset,
            index=index,
            meta_train=meta_train,
            query_text=text,
            top_k=k,
            query_id=sid,
        )

        retrieved_labels = [int(r["label"]) for r in retrieved]
        match_count = 0
        for lb in retrieved_labels:
            if int(lb) == int(y):
                match_count += 1

        match_frac = float(match_count / k) if k > 0 else 0.0

        per_sample.append(
            {
                "id": sid,
                "true_label": int(y),
                "k": int(k),
                "match_count": int(match_count),
                "match_frac": float(match_frac),
                "retrieved_ids": [str(r["id"]) for r in retrieved],
                "retrieved_labels": retrieved_labels,
            }
        )

    stats = compute_match_stats(per_sample, k)

    model = SentenceTransformer(config.rag_model, device=config.rag_device)
    train_emb = embed_texts(model, train_texts, config.rag_batch_size)
    test_emb = embed_texts(model, test_texts, config.rag_batch_size)

    all_emb = np.concatenate([train_emb, test_emb], axis=0)
    all_split = (["train"] * len(train_emb)) + (["test"] * len(test_emb))
    all_label = train_labels + test_labels
    all_ids = train_ids + test_ids

    z = make_2d_tsne(
        vectors=all_emb,
        seed=config.seed,
        perplexity=config.tsne_perplexity,
        n_iter=config.tsne_n_iter,
        pca_dim=config.pca_dim_before_tsne,
    )

    dataset_disp = dataset_display_name(dataset)

    out_dir = Path(config.runs_analysis_dir) / dataset.name / f"rag_match_k{k}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- label purity 저장 ---
    save_per_sample_csv(out_dir / "per_sample_match.csv", per_sample)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    plot_match_bar(out_dir / "match_bar.png", dataset_disp, k, stats)

    save_points_csv(out_dir / "tsne_points.csv", z, all_split, all_label, all_ids)
    plot_scatter(
        out_path=out_dir / "tsne_scatter.png",
        points=z,
        split=all_split,
        label=all_label,
        title=f"{dataset_disp} | {config.rag_model} | t-SNE(train/test, label)",
    )

    np.save(out_dir / "train_emb.npy", train_emb)
    np.save(out_dir / "test_emb.npy", test_emb)

    # --- boundary analysis 추가 ---
    train_signed, train_abs = compute_signed_distance_to_boundary(
        train_emb=train_emb,
        train_labels=train_labels,
        emb=train_emb,
        seed=config.seed,
    )
    test_signed, test_abs = compute_signed_distance_to_boundary(
        train_emb=train_emb,
        train_labels=train_labels,
        emb=test_emb,
        seed=config.seed,
    )

    # threshold는 고정값보다 "데이터 기반"이 비교에 더 안전합니다.
    # 예: train abs distance의 분위수(10%, 20%, 30%)를 "경계 근처" 기준으로 사용
    q10 = float(np.quantile(train_abs, 0.10)) if train_abs.size > 0 else 0.0
    q20 = float(np.quantile(train_abs, 0.20)) if train_abs.size > 0 else 0.0
    q30 = float(np.quantile(train_abs, 0.30)) if train_abs.size > 0 else 0.0
    thresholds = [q10, q20, q30]

    train_boundary = boundary_summary(train_abs, train_labels, thresholds)
    test_boundary = boundary_summary(test_abs, test_labels, thresholds)

    boundary_out = {
        "boundary_model": "logistic_regression_linear",
        "thresholds_source": "quantiles_of_train_abs_distance",
        "threshold_quantiles": [0.10, 0.20, 0.30],
        "thresholds_abs_distance": thresholds,
        "train": train_boundary,
        "test": test_boundary,
    }

    with (out_dir / "boundary_summary.json").open("w", encoding="utf-8") as f:
        json.dump(boundary_out, f, ensure_ascii=False, indent=2)

    save_boundary_per_sample_csv(
        out_path=out_dir / "boundary_train_per_sample.csv",
        ids=train_ids,
        labels=train_labels,
        signed_dist=train_signed,
        abs_dist=train_abs,
    )
    save_boundary_per_sample_csv(
        out_path=out_dir / "boundary_test_per_sample.csv",
        ids=test_ids,
        labels=test_labels,
        signed_dist=test_signed,
        abs_dist=test_abs,
    )

    plot_abs_distance_hist(
        out_path=out_dir / "boundary_abs_distance_hist_train.png",
        dataset_disp=dataset_disp,
        split_name="train",
        abs_dist=train_abs,
        labels=train_labels,
    )
    plot_abs_distance_hist(
        out_path=out_dir / "boundary_abs_distance_hist_test.png",
        dataset_disp=dataset_disp,
        split_name="test",
        abs_dist=test_abs,
        labels=test_labels,
    )

    print(f"[DONE] {dataset.name}: saved to {out_dir}")


def main():
    config = Config()
    set_seeds(config.seed)

    for dataset in TARGET_DATASETS:
        print(f"[RUN] dataset={dataset.name}")
        run_one_dataset(config, dataset)

    print("[DONE] all datasets completed.")


if __name__ == "__main__":
    main()
