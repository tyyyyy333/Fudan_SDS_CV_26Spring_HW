import numpy as np
import itertools
import pandas as pd
import random
import os
import json

import model as model_module
import visualization
import layer
import optim

from itertools import product


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def train_valid_split(X, y, valid_size=0.2, random_seed=42):
    np.random.seed(random_seed)
    permutation = np.random.permutation(len(X))
    X = X[permutation]
    y = y[permutation]
    split_index = int(len(X) * (1 - valid_size))
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]
    
    return X_train, y_train, X_val, y_val

def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

def one_hot_decode(labels):
    return np.argmax(labels, axis=1)

def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))

def confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def grid_search(param_grid, objective_fn, score_key="score", maximize=True):
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = []
    best_score = None
    best_config = None

    for i, config in enumerate(combinations):
        print(f"进度: [{i + 1}/{len(combinations)}] 测试参数: {config}")
        outcome = objective_fn(config)
        if isinstance(outcome, dict):
            score = float(outcome[score_key])
            record = {**config, **outcome}
        else:
            score = float(outcome)
            record = {**config, score_key: score}

        is_better = (
            best_score is None
            or (maximize and score > best_score)
            or ((not maximize) and score < best_score)
        )
        if is_better:
            best_score = score
            best_config = config.copy()

        results.append(record)

    df_results = pd.DataFrame(results).sort_values(
        by=score_key, ascending=not maximize
    )
    return {
        "results": df_results,
        "best_config": best_config,
        f"best_{score_key}": best_score,
    }

def _softmax_from_logits(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def _l2_penalty(net, l2_lambda):
    if l2_lambda <= 0:
        return 0.0
    penalty = 0.0
    for name, param, _ in net.named_parameters():
        if name.endswith(".weight"):
            penalty += float(np.sum(param ** 2))
    return 0.5 * l2_lambda * penalty


def _count_parameters(net):
    total = 0
    for _, param, _ in net.named_parameters():
        total += int(np.prod(param.shape))
    return total


def _safe_name(text):
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def _make_output_dir(args, exp_name):
    out_dir = os.path.join(args.output_dir, args.exp_name, exp_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _serialize_config(config):
    serial = {}
    for k, v in config.items():
        if isinstance(v, (list, tuple)):
            serial[k] = list(v)
        elif isinstance(v, np.generic):
            serial[k] = v.item()
        else:
            serial[k] = v
    return serial


def _baseline_config_from_args(args):
    return {
        "preprocess_mode": args.preprocess_mode,
        "hidden_sizes": list(args.hidden_sizes),
        "activation": args.activation,
        "dropout": args.dropout,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "lr_decay": args.lr_decay,
        "weight_decay": args.weight_decay,
    }

def _build_search_candidates():
    search_space = {
        "preprocess_mode": ["baseline", "flip", "mask", "flip_mask"],
        "hidden_sizes": [[128], [256], [512]],
        "activation": ["relu", "tanh", "sigmoid"],
        "dropout": [0.0, 0.2],
        "optimizer": ["sgd", "adam"],
        "learning_rate": [0.01, 0.001],
        "lr_decay": [0.95, 0.85],
        "weight_decay": [1e-4, 5e-4]
    }
    keys = search_space.keys()
    values = search_space.values()

    candidates = []
    for instance in product(*values):
        cfg = dict(zip(keys, instance))
        candidates.append(cfg)
        
    random.shuffle(candidates) 
    return candidates



def _choose_trial_configs(args, candidates):
    n = len(candidates)
    if args.max_trials <= 0 or args.max_trials >= n:
        if args.search_mode == "random":
            rng = np.random.default_rng(args.seed)
            idx = rng.permutation(n)
            return [candidates[i] for i in idx]
        
        return candidates

    if args.search_mode == "random":
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(n)[: args.max_trials]
        
        return [candidates[i] for i in idx]

    idx = np.linspace(0, n - 1, args.max_trials, dtype=int)
    idx = np.unique(idx)
    
    return [candidates[i] for i in idx]

def _build_model(args, config):
    hidden = config["hidden_sizes"]
    sizes = [28 * 28] + hidden + [10]
    net = model_module.MLP(
        size_list=sizes,
        activation=config["activation"],
        dropout=config["dropout"],
    )
    if config["optimizer"] == "sgd":
        optimizer = optim.SGD(
            net,
            lr=config["learning_rate"],
            momentum=args.momentum,
            weight_decay=config["weight_decay"],
        )
    else:
        optimizer = optim.Adam(
            net,
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
    scheduler = optim.ExponentialLR(optimizer, gamma=config["lr_decay"])
    criterion = layer.CrossEntropy()
    return net, optimizer, scheduler, criterion


def _top_confusions(cm, top_k):
    cm = np.asarray(cm)
    cm2 = cm.copy()
    np.fill_diagonal(cm2, 0)
    flat = np.argsort(cm2.ravel())[::-1]
    out = []
    total = cm.sum()
    for idx in flat:
        count = int(cm2.ravel()[idx])
        if count <= 0:
            break
        t = idx // cm.shape[1]
        p = idx % cm.shape[1]
        ratio = count / max(1, total)
        out.append((int(t), int(p), count, ratio))
        if len(out) >= top_k:
            break
    return out


def _write_search_report(path, results_df, best_result, baseline_result, top_k):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 参数搜索实现报告\n\n")

        f.write("## 实验设置\n\n")
        f.write("- 数据预处理：baseline / flip / mask / flip_mask\n")
        f.write("- 隐藏层结构：[[128],[256,128],[512,256]]\n")
        f.write("- 激活函数：ReLU / Tanh / Sigmoid\n")
        f.write("- Dropout：0.0 / 0.2\n")
        f.write("- 优化器：SGD / Adam\n")
        f.write("- 学习率：(0.01,0.001)\n")
        f.write("- 学习率衰减：(0.95,0.9)\n")
        f.write("- 权重衰减：(1e-4,5e-4)\n")

        f.write("## 结果\n\n")
        f.write(f"- 试验总数：{len(results_df)}\n")
        f.write(f"- 最佳验证准确率：{best_result['best_val_acc']:.4f}\n")
        if best_result["test_acc"] is not None:
            f.write(f"- 最佳配置测试准确率：{best_result['test_acc']:.4f}\n")
        f.write(f"- 基线配置测试准确率：{baseline_result['test_acc']:.4f}\n\n")

        f.write("### Top-K 配置\n\n")
        f.write(results_df.head(top_k).to_markdown(index=False))
        f.write("\n\n")

        f.write("### 最优与基线差异\n\n")
        best_cfg = best_result["config"]
        base_cfg = baseline_result["config"]
        keys = [
            "preprocess_mode",
            "hidden_sizes",
            "activation",
            "dropout",
            "optimizer",
            "learning_rate",
            "lr_decay",
            "weight_decay",
        ]
        for k in keys:
            f.write(f"- {k}: baseline={base_cfg[k]} -> best={best_cfg[k]}\n")
        f.write("\n")

        if best_result.get("top_confusions"):
            f.write("## 错误分析\n\n")
            for t, p, c, r in best_result["top_confusions"]:
                f.write(
                    f"- `{CLASS_NAMES[t]}` -> `{CLASS_NAMES[p]}`: {c} 次（占全部样本 {r:.2%}）\n"
                )
            f.write("\n")


def _write_json(path, content):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)


def _save_single_experiment_artifacts(out_dir, result, save_plots):
    _write_json(os.path.join(out_dir, "run_summary.json"), result["summary"])
    _write_json(os.path.join(out_dir, "train_history.json"), result["history"])
    if save_plots:
        visualization.plot_training_curves(
            result["history"], save_path=os.path.join(out_dir, "training_curves.png")
        )
        first_layer_w = result["model"].get_first_layer_weights()
        if first_layer_w is not None:
            visualization.visualize_first_layer_weights(
                first_layer_w,
                image_shape=(28, 28),
                save_path=os.path.join(out_dir, "first_layer_weights.png"),
            )
        if result["test_cm"] is not None:
            visualization.plot_confusion_matrix(
                result["test_cm"],
                class_names=CLASS_NAMES,
                normalize=False,
                save_path=os.path.join(out_dir, "confusion_matrix_count.png"),
            )
            visualization.plot_confusion_matrix(
                result["test_cm"],
                class_names=CLASS_NAMES,
                normalize=True,
                save_path=os.path.join(out_dir, "confusion_matrix_norm.png"),
            )
