import math
import numpy as np
import matplotlib.pyplot as plt


def plot_training_curves(history, save_path=None):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], label="Val Acc")
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_training_comparison(history_a, history_b, label_a="A", label_b="B", save_path=None):
    epochs_a = np.arange(1, len(history_a["train_loss"]) + 1)
    epochs_b = np.arange(1, len(history_b["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs_a, history_a["val_loss"], label=f"{label_a} Val Loss")
    axes[0].plot(epochs_b, history_b["val_loss"], label=f"{label_b} Val Loss")
    axes[0].set_title("Validation Loss Comparison")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs_a, history_a["val_acc"], label=f"{label_a} Val Acc")
    axes[1].plot(epochs_b, history_b["val_acc"], label=f"{label_b} Val Acc")
    axes[1].set_title("Validation Accuracy Comparison")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_confusion_matrix(cm, class_names=None, normalize=False, save_path=None):
    cm = np.asarray(cm, dtype=np.float64)
    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True) + 1e-12
        cm = cm / row_sum

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def visualize_first_layer_weights(weight_matrix, image_shape=(28, 28), max_show=64, save_path=None):
    W = np.asarray(weight_matrix)
    if W.ndim != 2:
        raise ValueError("weight_matrix should be 2D with shape [input_dim, hidden_dim].")
    input_dim, hidden_dim = W.shape
    if input_dim != image_shape[0] * image_shape[1]:
        raise ValueError("input_dim does not match image_shape.")

    show_n = min(max_show, hidden_dim)
    grid_cols = int(np.ceil(np.sqrt(show_n)))
    grid_rows = int(np.ceil(show_n / grid_cols))

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 1.4, grid_rows * 1.4))
    axes = np.array(axes).reshape(-1)

    for i in range(grid_rows * grid_cols):
        ax = axes[i]
        ax.axis("off")
        if i < show_n:
            w_img = W[:, i].reshape(image_shape)
            ax.imshow(w_img, cmap="seismic")

    fig.suptitle("First Layer Weights", y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_error_samples(X, y_true, y_pred, max_show=16, image_shape=(28, 28), save_path=None):
    X = np.asarray(X)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    wrong_idx = np.where(y_true != y_pred)[0]
    if len(wrong_idx) == 0:
        return None

    show_n = min(max_show, len(wrong_idx))
    grid_cols = int(math.ceil(math.sqrt(show_n)))
    grid_rows = int(math.ceil(show_n / grid_cols))

    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 2.0, grid_rows * 2.0))
    axes = np.array(axes).reshape(-1)

    for i in range(grid_rows * grid_cols):
        ax = axes[i]
        ax.axis("off")
        if i < show_n:
            idx = wrong_idx[i]
            ax.imshow(X[idx].reshape(image_shape), cmap="gray")
            ax.set_title(f"T:{y_true[idx]} P:{y_pred[idx]}", fontsize=9)

    fig.suptitle("Misclassified Samples", y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_grouped_error_samples(
    X,
    y_true,
    y_pred,
    confusion_pairs,
    class_names=None,
    per_group=4,
    image_shape=(28, 28),
    save_path=None,
):
    X = np.asarray(X)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if class_names is None:
        class_names = [str(i) for i in range(10)]

    collected = []
    titles = []
    for t, p in confusion_pairs:
        idx = np.where((y_true == t) & (y_pred == p))[0][:per_group]
        for i in idx:
            collected.append(i)
            titles.append(f"T:{class_names[t]} -> P:{class_names[p]}")

    if len(collected) == 0:
        return None

    show_n = len(collected)
    grid_cols = min(4, show_n)
    grid_rows = int(math.ceil(show_n / grid_cols))
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * 3.0, grid_rows * 3.0))
    axes = np.array(axes).reshape(-1)

    for i in range(grid_rows * grid_cols):
        ax = axes[i]
        ax.axis("off")
        if i < show_n:
            idx = collected[i]
            ax.imshow(X[idx].reshape(image_shape), cmap="gray")
            ax.set_title(titles[i], fontsize=8)

    fig.suptitle("Grouped Misclassified Samples", y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig
