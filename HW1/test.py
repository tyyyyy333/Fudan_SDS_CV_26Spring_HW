import argparse
import os

import numpy as np
import pandas as pd

import data_process
import layer
import model as model_module
import train
import utils
import visualization


CLASS_NAMES = train.CLASS_NAMES


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Test trained MLP on Fashion-MNIST.")
    parser.add_argument("--dataset", type=str, default="fashion_mnist")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--model_path", type=str, default="output/exp_grid/best/best_model.npz")
    parser.add_argument("--preprocess_mode", type=str, default="baseline", choices=["baseline", "flip", "mask", "flip_mask"])

    parser.add_argument("--exp_name", type=str, default="exp_main")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--plot_cm", action="store_true")
    parser.add_argument("--plot_errors", action="store_true")
    parser.add_argument("--num_error_show", type=int, default=16)
    parser.add_argument("--save_report", action="store_true")
    parser.add_argument("--top_confusions_k", type=int, default=5)
    parser.add_argument("--error_grouped", action="store_true")
    parser.add_argument("--save_error_table", action="store_true")
    return parser


def build_model(args):
    sizes = [28 * 28] + args.hidden_sizes + [10]
    net = model_module.MLP(
        size_list=sizes,
        activation=args.activation,
        dropout=args.dropout,
    )
    net.load(args.model_path)
    return net


def top_confusions(cm, top_k):
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
        out.append((int(t), int(p), count, count / max(1, total)))
        if len(out) >= top_k:
            break
    return out


def reason_hint(t, p):
    tops = {0, 2, 3, 4, 6}
    shoes = {5, 7, 9}
    if t in tops and p in tops:
        return "上衣类轮廓接近，边缘与纹理差异弱"
    if t in shoes and p in shoes:
        return "鞋类形状相似，视角和边缘细节易混淆"
    return "局部形状和灰度分布重叠，样本可分性不足"


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    utils.set_seed(args.seed)

    out_dir = os.path.join(args.output_dir, args.exp_name, "test")
    os.makedirs(out_dir, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = train.prepare_data(
        args, preprocess_mode=args.preprocess_mode
    )
    _ = (X_train, y_train, X_val, y_val)
    test_loader = data_process.DataLoader(
        data_process.MnistDataset(X_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
    )

    net = build_model(args)
    criterion = layer.CrossEntropy()
    test_loss, test_acc, test_preds, test_targets = train.evaluate(
        net, test_loader, criterion, l2_lambda=0.0
    )
    cm = utils.confusion_matrix(test_targets, test_preds, num_classes=10)
    top_pairs = top_confusions(cm, args.top_confusions_k)

    print(f"Model: {args.model_path}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    if args.plot_cm:
        visualization.plot_confusion_matrix(
            cm,
            class_names=CLASS_NAMES,
            normalize=False,
            save_path=os.path.join(out_dir, "confusion_matrix_count.png"),
        )
        visualization.plot_confusion_matrix(
            cm,
            class_names=CLASS_NAMES,
            normalize=True,
            save_path=os.path.join(out_dir, "confusion_matrix_norm.png"),
        )

    if args.plot_errors:
        visualization.plot_error_samples(
            X_test,
            test_targets,
            test_preds,
            max_show=args.num_error_show,
            image_shape=(28, 28),
            save_path=os.path.join(out_dir, "error_samples.png"),
        )

    if args.error_grouped and len(top_pairs) > 0:
        pair_list = [(t, p) for t, p, _, _ in top_pairs]
        visualization.plot_grouped_error_samples(
            X_test,
            test_targets,
            test_preds,
            confusion_pairs=pair_list,
            class_names=CLASS_NAMES,
            per_group=3,
            image_shape=(28, 28),
            save_path=os.path.join(out_dir, "error_samples_grouped.png"),
        )

    if args.save_error_table:
        rows = []
        for t, p, c, r in top_pairs:
            rows.append(
                {
                    "true_class": CLASS_NAMES[t],
                    "pred_class": CLASS_NAMES[p],
                    "count": c,
                    "ratio_all_samples": r,
                    "reason_hint": reason_hint(t, p),
                }
            )
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "top_confusions.csv"), index=False)

    if args.save_report:
        report_path = os.path.join(out_dir, "test_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Test Report\n\n")
            f.write(f"- Model: `{args.model_path}`\n")
            f.write(f"- Preprocess mode: `{args.preprocess_mode}`\n")
            f.write(f"- Test loss: `{test_loss:.4f}`\n")
            f.write(f"- Test accuracy: `{test_acc:.4f}`\n\n")
            f.write("## Top Confusions\n\n")
            if top_pairs:
                for t, p, n, r in top_pairs:
                    f.write(
                        f"- `{CLASS_NAMES[t]}` -> `{CLASS_NAMES[p]}`: `{n}` 次，"
                        f"占全体样本 `{r:.2%}`，原因：{reason_hint(t, p)}\n"
                    )
            else:
                f.write("- 无明显错分。\n")
            f.write("\n## Error Analysis Notes\n\n")
            f.write("- 高频错分集中在相似轮廓类别，说明模型更多依赖粗粒度形状。\n")
            f.write("- 可通过更强数据增强和更丰富特征表达进一步缓解。\n")


if __name__ == "__main__":
    main()
