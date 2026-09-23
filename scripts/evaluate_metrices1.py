import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import traceback

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

try:
    from models import ConvNeXtVisionMambaTabNet
    print("✓ Model imported successfully")
except ImportError as e:
    print(f"✗ Failed to import model: {e}")
    exit(1)

# ============================================
# CONFIG (MEMORY‑SAFE)
# ============================================
ROOT_DIR = r"D:\liver\CViMXAI"
DATA_DIR = os.path.join(ROOT_DIR, "data")
WEIGHTS_DIR = os.path.join(ROOT_DIR, "weights")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Memory settings for 4GB GPU
BATCH_SIZE = 1                # safe for 4GB
NUM_FOLDS = 5
RANDOM_STATE = 42
USE_HALF_PRECISION = False    # disabled – avoids cuBLAS initialisation issues

# Warm‑up function to initialise cuBLAS before first real forward
def warmup_cublas(model, device):
    print("Warming up cuBLAS...")
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    if USE_HALF_PRECISION:
        dummy_input = dummy_input.half()
        model = model.half()
    with torch.inference_mode():
        _ = model(dummy_input)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    print("Warm‑up complete.")

DATASET_CONFIGS = {
    "CXRay": {
        "root": os.path.join(DATA_DIR, "Covid-19_Chest_XRay"),
        "classes": ["COVID", "NORMAL", "PNEUMONIA"],
        "num_classes": 3,
        "cm_title": "CXR1 - Confusion Matrix",
        "roc_title": "CXR1 - ROC Curves"
    },
    "Covid19_Ultrasound": {
        "root": os.path.join(DATA_DIR, "Covid19_Ultrasound"),
        "classes": ["covid", "normal", "pneumonia"],
        "num_classes": 3,
        "cm_title": "Ultrasound - Confusion Matrix",
        "roc_title": "Ultrasound - ROC Curves"
    },
    "Lung_XRay": {
        "root": os.path.join(DATA_DIR, "Lung_XRay"),
        "classes": ["Lung_Opacity", "Normal", "Viral_Pneumonia"],
        "num_classes": 3,
        "cm_title": "CXR2 - Confusion Matrix ",
        "roc_title": "CXR2 - ROC Curves"
    }
}

# Transform (no normalisation – model may handle it)
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def get_dataset(dataset_root):
    return datasets.ImageFolder(root=dataset_root, transform=eval_transform)

def get_fold_loaders(dataset_root, batch_size=1, num_folds=5):
    dataset = get_dataset(dataset_root)
    labels = np.array(dataset.targets)
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=RANDOM_STATE)
    fold_loaders = []
    for fold_idx, (_, val_idx) in enumerate(kf.split(range(len(dataset))), start=1):
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
        fold_loaders.append((fold_idx, val_loader))
    return fold_loaders

def evaluate_one_fold(model, loader, device, use_half=False):
    model.eval()
    if use_half:
        model = model.half()
    all_labels, all_preds, all_probs = [], [], []
    with torch.inference_mode():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            if use_half:
                imgs = imgs.half()
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_percent = (cm.astype('float') / row_sums) * 100
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_percent, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    plt.yticks(np.arange(len(class_names)), class_names)
    thresh = cm_percent.max() / 2.0 if cm_percent.max() > 0 else 50
    for i in range(cm_percent.shape[0]):
        for j in range(cm_percent.shape[1]):
            plt.text(j, i, f"{cm_percent[i, j]:.1f}%",
                     ha="center", color="white" if cm_percent[i, j] > thresh else "black")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_multiclass_roc(y_true, y_prob, num_classes, class_names, save_path, title="ROC Curve"):
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    fpr, tpr, roc_auc = {}, {}, {}
    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], lw=2, label=f"{class_names[i]} (AUC = {roc_auc[i]:.4f})")
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.plot(fpr["micro"], tpr["micro"], linestyle="--", lw=2, label=f"micro-average (AUC = {roc_auc['micro']:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return roc_auc

def evaluate_dataset(dataset_name, config):
    print(f"\n========== Evaluating {dataset_name} ==========")
    try:
        dataset_root = config["root"]
        classes = config["classes"]
        num_classes = config["num_classes"]
        cm_title = config.get("cm_title", f"{dataset_name} Confusion Matrix (%)")
        roc_title = config.get("roc_title", f"{dataset_name} ROC Curve")

        weight_dir = os.path.join(WEIGHTS_DIR, dataset_name)
        save_dir = os.path.join(RESULTS_DIR, dataset_name, "evaluation")
        os.makedirs(save_dir, exist_ok=True)

        fold_loaders = get_fold_loaders(dataset_root, batch_size=BATCH_SIZE, num_folds=NUM_FOLDS)

        all_labels_all_folds = []
        all_preds_all_folds = []
        all_probs_all_folds = []
        fold_metrics = []

        # Warm‑up once per dataset (initialises cuBLAS)
        warmup_model = ConvNeXtVisionMambaTabNet(num_classes=num_classes).to(DEVICE)
        warmup_cublas(warmup_model, DEVICE)
        del warmup_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for fold, val_loader in fold_loaders:
            print(f"Evaluating fold {fold}...")
            weight_path = os.path.join(weight_dir, f"best_model_fold_{fold}.pth")
            if not os.path.exists(weight_path):
                print(f"Skipping fold {fold}. Weight not found: {weight_path}")
                continue

            model = ConvNeXtVisionMambaTabNet(num_classes=num_classes).to(DEVICE)
            model.load_state_dict(torch.load(weight_path, map_location=DEVICE))

            # Optional: set model to half only if USE_HALF_PRECISION is True
            if USE_HALF_PRECISION:
                model = model.half()

            y_true, y_pred, y_prob = evaluate_one_fold(
                model, val_loader, DEVICE, use_half=USE_HALF_PRECISION
            )

            del model
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            all_labels_all_folds.extend(y_true.tolist())
            all_preds_all_folds.extend(y_pred.tolist())
            all_probs_all_folds.extend(y_prob.tolist())

            fold_acc = accuracy_score(y_true, y_pred)
            fold_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
            fold_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
            fold_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
            fold_metrics.append({
                "fold": fold,
                "accuracy": float(fold_acc),
                "precision": float(fold_prec),
                "recall": float(fold_rec),
                "f1_score": float(fold_f1)
            })

        if len(all_labels_all_folds) == 0:
            print(f"No predictions available for {dataset_name}.")
            return

        all_labels_all_folds = np.array(all_labels_all_folds)
        all_preds_all_folds = np.array(all_preds_all_folds)
        all_probs_all_folds = np.array(all_probs_all_folds)

        overall_acc = accuracy_score(all_labels_all_folds, all_preds_all_folds)
        overall_prec = precision_score(all_labels_all_folds, all_preds_all_folds, average="macro", zero_division=0)
        overall_rec = recall_score(all_labels_all_folds, all_preds_all_folds, average="macro", zero_division=0)
        overall_f1 = f1_score(all_labels_all_folds, all_preds_all_folds, average="macro", zero_division=0)

        cm = confusion_matrix(all_labels_all_folds, all_preds_all_folds)

        report = classification_report(
            all_labels_all_folds, all_preds_all_folds,
            target_names=classes, digits=4, zero_division=0, output_dict=True
        )

        roc_auc_dict = plot_multiclass_roc(
            y_true=all_labels_all_folds, y_prob=all_probs_all_folds,
            num_classes=num_classes, class_names=classes,
            save_path=os.path.join(save_dir, "roc_curve.png"), title=roc_title
        )

        plot_confusion_matrix(
            cm=cm, class_names=classes,
            save_path=os.path.join(save_dir, "confusion_matrix.png"), title=cm_title
        )

        summary = {
            "dataset": dataset_name,
            "overall_accuracy": float(overall_acc),
            "overall_precision_macro": float(overall_prec),
            "overall_recall_macro": float(overall_rec),
            "overall_f1_macro": float(overall_f1),
            "roc_auc": {str(k): float(v) for k, v in roc_auc_dict.items()},
            "fold_metrics": fold_metrics,
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        }

        with open(os.path.join(save_dir, "evaluation_summary.json"), "w") as f:
            json.dump(summary, f, indent=4)

        print(f"Saved evaluation results for {dataset_name} at: {save_dir}")

    except Exception as e:
        print(f"\n❌ ERROR in {dataset_name}:")
        print(f"{type(e).__name__}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    for dataset_name, config in DATASET_CONFIGS.items():
        evaluate_dataset(dataset_name, config)
    print("\nAll evaluations completed successfully.")