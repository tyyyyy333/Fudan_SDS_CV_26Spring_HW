import argparse
import copy
import json
import os

from tqdm import tqdm
import numpy as np
import pandas as pd

import data_process
import layer
import model as model_module
import optim
import utils
import visualization


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

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train MLP on Fashion-MNIST.")
    parser.add_argument("--dataset", type=str, default="fashion_mnist")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_decay", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[256])
    parser.add_argument("--preprocess_mode", type=str, default="baseline", choices=["baseline", "flip", "mask", "flip_mask"])
    parser.add_argument("--valid_size", type=float, default=0.1)

    parser.add_argument("--grid_search", action="store_true")
    parser.add_argument("--search_mode", type=str, default="grid", choices=["grid", "random"])
    parser.add_argument("--max_trials", type=int, default=18)
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--top_k", type=int, default=5)

    parser.add_argument("--exp_name", type=str, default="exp_main")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument("--save_path", type=str, default="MWeight/best_model.npz")
    return parser


def train_one_epoch(net, loader, criterion, optimizer):
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        y_onehot = utils.one_hot_encode(y_batch, num_classes=10).astype(np.float32)
        logits = net.forward(X_batch, training=True)
        probs = utils._softmax_from_logits(logits)
        ce_loss = criterion.forward(logits, y_onehot)
        l2_loss = utils._l2_penalty(net, optimizer.weight_decay)
        loss = ce_loss + l2_loss

        grad = criterion.backward()
        net.zero_grad()
        net.backward(grad)
        optimizer.step()

        preds = np.argmax(probs, axis=1)
        correct += np.sum(preds == y_batch)
        total += y_batch.shape[0]
        total_loss += loss * y_batch.shape[0]

    return total_loss / total, correct / total


def evaluate(net, loader, criterion, l2_lambda=0.0):
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    for X_batch, y_batch in loader:
        y_onehot = utils.one_hot_encode(y_batch, num_classes=10).astype(np.float32)
        logits = net.forward(X_batch, training=False)
        probs = utils._softmax_from_logits(logits)
        loss = criterion.forward(logits, y_onehot) + utils._l2_penalty(net, l2_lambda)
        preds = np.argmax(probs, axis=1)

        correct += np.sum(preds == y_batch)
        total += y_batch.shape[0]
        total_loss += loss * y_batch.shape[0]
        all_preds.append(preds)
        all_targets.append(y_batch)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    return total_loss / total, correct / total, all_preds, all_targets


def train_model(net, optimizer, scheduler, criterion, train_loader, val_loader, epochs):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_state = None
    best_epoch = -1

    for epoch in tqdm(range(1, epochs + 1)):
        train_loss, train_acc = train_one_epoch(net, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(
            net, val_loader, criterion, l2_lambda=optimizer.weight_decay
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(net.state_dict())
            best_epoch = epoch

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={optimizer.lr:.6f}"
        )

    if best_state is not None:
        net.load_state_dict(best_state)
    return net, history, best_val_acc, best_epoch


def build_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size):
    train_loader = data_process.DataLoader(
        data_process.MnistDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = data_process.DataLoader(
        data_process.MnistDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = data_process.DataLoader(
        data_process.MnistDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader


def prepare_data(args, preprocess_mode=None):
    
    mode = preprocess_mode if preprocess_mode is not None else args.preprocess_mode
    valid_size = getattr(args, "valid_size", 0.1)
    
    X_train_all, y_train_all, X_test, y_test = data_process.build_numpy_splits(
        dataset_name=args.dataset,
        preprocess_mode=mode,
        seed=args.seed,
    )
    
    X_train, y_train, X_val, y_val = utils.train_valid_split(
        X_train_all,
        y_train_all,
        valid_size=valid_size,
        random_seed=args.seed,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test

def run_single_experiment(
    args,
    config,
    prepared_data=None,
    evaluate_test=True,
    exp_name="single_run",
    save_artifacts=True,
):
    net, optimizer, scheduler, criterion = utils._build_model(args, config)
    if prepared_data is None:
        prepared_data = prepare_data(args, preprocess_mode=config["preprocess_mode"])
    X_train, y_train, X_val, y_val, X_test, y_test = prepared_data
    train_loader, val_loader, test_loader = build_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, args.batch_size
    )

    net, history, best_val_acc, best_epoch = train_model(
        net,
        optimizer,
        scheduler,
        criterion,
        train_loader,
        val_loader,
        args.epochs,
    )
    val_loss, val_acc, _, _ = evaluate(
        net, val_loader, criterion, l2_lambda=optimizer.weight_decay
    )

    test_loss = None
    test_acc = None
    test_cm = None
    test_preds = None
    test_targets = None
    if evaluate_test:
        test_loss, test_acc, test_preds, test_targets = evaluate(
            net, test_loader, criterion, l2_lambda=optimizer.weight_decay
        )
        test_cm = utils.confusion_matrix(test_targets, test_preds, num_classes=10)

    summary = {
        "config": utils._serialize_config(config),
        "best_epoch": best_epoch,
        "best_val_acc": float(best_val_acc),
        "final_val_acc": float(val_acc),
        "final_val_loss": float(val_loss),
        "test_acc": None if test_acc is None else float(test_acc),
        "test_loss": None if test_loss is None else float(test_loss),
        "param_count": utils._count_parameters(net),
    }

    result = {
        "history": history,
        "summary": summary,
        "model": net,
        "test_cm": test_cm,
        "test_preds": test_preds,
        "test_targets": test_targets,
        "X_test": X_test,
    }

    if save_artifacts:
        out_dir = utils._make_output_dir(args, exp_name)
        model_path = os.path.join(out_dir, "best_model.npz")
        net.save(model_path)
        summary["model_path"] = model_path
        utils._save_single_experiment_artifacts(out_dir, result, save_plots=args.save_plots)
        
    return result


def run_search(args):
    candidates = utils._build_search_candidates()
    trial_configs = utils._choose_trial_configs(args, candidates)
    data_cache = {}
    trial_records = []

    print(f"Search mode={args.search_mode}, total candidates={len(candidates)}, run trials={len(trial_configs)}")
    
    for i, cfg in enumerate(trial_configs, start=1):
        print()
        print(f"\n[Trial {i}/{len(trial_configs)}] {cfg}")
        
        mode = cfg["preprocess_mode"]
        if mode not in data_cache:
            data_cache[mode] = prepare_data(args, preprocess_mode=mode)
            
        result = run_single_experiment(
            args,
            config=cfg,
            prepared_data=data_cache[mode],
            evaluate_test=False,
            exp_name=f"trial_{i:03d}",
            save_artifacts=False,
        )
        
        trial_records.append(
            {
                "trial_id": i,
                **utils._serialize_config(cfg),
                "best_val_acc": result["summary"]["best_val_acc"],
                "best_epoch": result["summary"]["best_epoch"],
                "param_count": result["summary"]["param_count"],
            }
        )

    results_df = pd.DataFrame(trial_records).sort_values(by="best_val_acc", ascending=False)
    best_cfg = trial_configs[int(results_df.iloc[0]["trial_id"]) - 1]

    baseline_cfg = utils._baseline_config_from_args(args)
    
    if baseline_cfg["preprocess_mode"] not in data_cache:
        data_cache[baseline_cfg["preprocess_mode"]] = prepare_data(
            args, preprocess_mode=baseline_cfg["preprocess_mode"]
        )
        
    baseline_result = run_single_experiment(
        args,
        config=baseline_cfg,
        prepared_data=data_cache[baseline_cfg["preprocess_mode"]],
        evaluate_test=True,
        exp_name="baseline",
        save_artifacts=True,
    )

    best_result = run_single_experiment(
        args,
        config=best_cfg,
        prepared_data=data_cache[best_cfg["preprocess_mode"]],
        evaluate_test=True,
        exp_name="best",
        save_artifacts=True,
    )

    best_cm = best_result["test_cm"]
    top_conf = utils._top_confusions(best_cm, args.top_k) if best_cm is not None else []
    best_result["top_confusions"] = top_conf

    root_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(root_dir, exist_ok=True)
    results_path = os.path.join(root_dir, "search_results.csv")
    results_df.to_csv(results_path, index=False)
    
    if args.save_csv:
        print(f"Search results CSV saved to: {results_path}")

    visualization.plot_training_comparison(
        baseline_result["history"],
        best_result["history"],
        label_a="baseline",
        label_b="best",
        save_path=os.path.join(root_dir, "training_comparison.png"),
    )

    utils._write_json(
        os.path.join(root_dir, "best_config.json"),
        {"best_config": utils._serialize_config(best_cfg), "top_confusions": top_conf},
    )
    utils._write_search_report(
        os.path.join(root_dir, "report_full.md"),
        results_df,
        {**best_result["summary"], "config": utils._serialize_config(best_cfg), "top_confusions": top_conf},
        {**baseline_result["summary"], "config": utils._serialize_config(baseline_cfg)},
        top_k=args.top_k,
    )
    print(f"Best config: {best_cfg}")


def run_single(args):
    cfg = utils._baseline_config_from_args(args)
    prepared_data = prepare_data(args, preprocess_mode=cfg["preprocess_mode"])
    result = run_single_experiment(
        args,
        config=cfg,
        prepared_data=prepared_data,
        evaluate_test=True,
        exp_name="single",
        save_artifacts=True,
    )
    print(f"Best validation accuracy: {result['summary']['best_val_acc']:.4f}")
    if result["summary"]["test_acc"] is not None:
        print(f"Test accuracy: {result['summary']['test_acc']:.4f}")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    utils.set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)

    if args.grid_search:
        run_search(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
