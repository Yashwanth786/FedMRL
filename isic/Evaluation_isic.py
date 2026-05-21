import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    precision_recall_curve
)

# ==========================================================
#   ASK USER WHICH MODEL TO EVALUATE (0 or 1)
# ==========================================================
def select_model_folder():
    print("\nSelect model type:")
    print("0 → saved_models")
    print("1 → saved_models_gnn")

    choice = input("Enter choice (0/1): ").strip()
    while choice not in ["0", "1"]:
        choice = input("Invalid input. Enter 0 or 1: ").strip()

    if choice == "0":
        print("📁 Using saved_models/")
        return "saved_models", "evaluation_report"
    else:
        print("📁 Using saved_models_gnn/")
        return "saved_models_gnn", "evaluation_report_gnn"


# ==========================================================
#         LOAD IMAGES/LABELS FROM test .NPY FILES
# ==========================================================
def load_numpy_dataset(images_npy, labels_npy):
    if not os.path.exists(images_npy):
        raise FileNotFoundError(images_npy)
    if not os.path.exists(labels_npy):
        raise FileNotFoundError(labels_npy)

    X = np.load(images_npy).astype("float32") / 255.0
    y_raw = np.load(labels_npy)

    if y_raw.ndim == 1:         # int labels
        y = y_raw.astype(int)
        num_classes = int(np.max(y)) + 1
        y_onehot = tf.keras.utils.to_categorical(y, num_classes)
    else:                       # one-hot labels
        y_onehot = y_raw.astype("float32")
        y = np.argmax(y_onehot, axis=1)
        num_classes = y_onehot.shape[1]

    print(f"Loaded test set: {len(X)} samples, {num_classes} classes")
    return X, y, y_onehot, num_classes


# ==========================================================
#               AUTO-LOAD TRAINED MODEL
# ==========================================================
def load_model_auto(model_dir, model_path=None):
    candidates = [
        model_path,
        os.path.join(model_dir, "best_model_full"),
        os.path.join(model_dir, "final_model_full"),
        os.path.join(model_dir, "best_model.h5"),
        os.path.join(model_dir, "model.h5"),
    ]

    for p in candidates:
        if p and os.path.exists(p):
            try:
                print(f"📦 Loading model from: {p}")
                return tf.keras.models.load_model(p)
            except Exception as e:
                print("⚠️ Load failed:", e)

    raise FileNotFoundError("❌ No valid model found.")


# ==========================================================
#                   METRIC CALCULATION
# ==========================================================
def evaluate(model, X, y, y_onehot):
    y_prob = model.predict(X, batch_size=32)
    y_pred = np.argmax(y_prob, axis=1)

    metrics = {
        "Accuracy (%)": accuracy_score(y, y_pred) * 100,
        "Precision (%)": precision_score(y, y_pred, average="weighted", zero_division=0) * 100,
        "Recall (%)": recall_score(y, y_pred, average="weighted", zero_division=0) * 100,
        "F1-score (%)": f1_score(y, y_pred, average="weighted", zero_division=0) * 100,
        "Balanced Accuracy (%)": balanced_accuracy_score(y, y_pred) * 100,
        "Kappa": cohen_kappa_score(y, y_pred),
        "MCC": matthews_corrcoef(y, y_pred),
        "Confusion Matrix": confusion_matrix(y, y_pred)
    }

    try:
        metrics["AUC (%)"] = roc_auc_score(y_onehot, y_prob, multi_class="ovo", average="weighted") * 100
    except:
        metrics["AUC (%)"] = None

    return metrics, y_pred, y_prob


# ==========================================================
#                       PLOTS
# ==========================================================
def save_confusion_matrix(cm, class_names, REPORT_DIR):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{REPORT_DIR}/confusion_matrix.png", dpi=200, bbox_inches='tight')
    plt.close()


def save_metric_barplot(metrics, REPORT_DIR):
    names = ["Accuracy", "Precision", "Recall", "F1-score", "AUC"]
    values = [
        metrics["Accuracy (%)"],
        metrics["Precision (%)"],
        metrics["Recall (%)"],
        metrics["F1-score (%)"],
        metrics["AUC (%)"] if metrics["AUC (%)"] else 0,
    ]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=names, y=values, palette="viridis")
    plt.ylabel("Percentage (%)")
    plt.title("Model Metrics")
    plt.ylim(0, 100)
    plt.savefig(f"{REPORT_DIR}/metrics_barplot.png", dpi=200)
    plt.close()


def save_roc_curves(y, y_prob, num_classes, REPORT_DIR):
    plt.figure(figsize=(9, 9))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve((y == i).astype(int), y_prob[:, i])
        plt.plot(fpr, tpr, label=f"Class {i}")

    plt.plot([0, 1], [0, 1], "k--")
    plt.title("ROC Curves")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.savefig(f"{REPORT_DIR}/roc_curves.png", dpi=200)
    plt.close()


def save_precision_recall_curves(y, y_prob, num_classes, REPORT_DIR):
    plt.figure(figsize=(7, 6))
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve((y == i).astype(int), y_prob[:, i])
        plt.plot(recall, precision, label=f"Class {i}")

    plt.title("Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.savefig(f"{REPORT_DIR}/precision_recall_curves.png", dpi=200)
    plt.close()
    
# ==========================================================
#                           MAIN
# ==========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_images", default="test_images.npy")
    parser.add_argument("--test_labels", default="test_labels.npy")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    # Model selection
    MODEL_DIR, REPORT_DIR = select_model_folder()
    os.makedirs(REPORT_DIR, exist_ok=True)

    # Load model
    model = load_model_auto(MODEL_DIR, args.model)

    # Load test data
    X, y, y_onehot, num_classes = load_numpy_dataset(
        args.test_images, args.test_labels
    )

    # Evaluate
    metrics, y_pred, y_prob = evaluate(model, X, y, y_onehot)

    # Save plots
    CLASS_NAMES = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]
    save_confusion_matrix(metrics["Confusion Matrix"], CLASS_NAMES, REPORT_DIR)
    save_metric_barplot(metrics, REPORT_DIR)
    save_roc_curves(y, y_prob, num_classes, REPORT_DIR)
    save_precision_recall_curves(y, y_prob, num_classes, REPORT_DIR)

    # Save summary
    with open(f"{REPORT_DIR}/metrics_summary.txt", "w") as f:
        for k, v in metrics.items():
            if k == "Confusion Matrix":
                continue
            f.write(f"{k}: {v}\n")

    print("\n🎉 Evaluation complete!")
    print(f"📁 Results saved in: {REPORT_DIR}")


if __name__ == "__main__":
    main()
