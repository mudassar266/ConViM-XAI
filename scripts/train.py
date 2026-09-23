# train.py

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models import ConvNeXtVisionMambaTabNet
from dataset_loader import get_dataloaders_for_fold

# ===========================
# CONFIG
# ===========================
EPOCHS = 30
BATCH_SIZE = 8
NUM_CLASSES = 3   # update if needed
DATASET_NAME = "Lung_XRay"  # change dynamically if needed

DEVICE = "cuda" #if torch.cuda.is_available() else "cpu"


# ===========================
# TRAIN ONE FOLD
# ===========================
def train_one_fold(model, train_loader, val_loader, fold):

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_acc = 0.0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": []
    }

    # Create directories
    weight_dir = f"weights/{DATASET_NAME}"
    result_dir = f"results/{DATASET_NAME}/fold_{fold}"

    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    for epoch in range(EPOCHS):

        # ===== TRAIN =====
        model.train()
        train_loss = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(acc)

        print(f"Fold {fold} | Epoch {epoch+1}/{EPOCHS} | Val Acc: {acc:.4f}")

        # ===== SAVE BEST MODEL =====
        if acc > best_acc:
            best_acc = acc

            torch.save(model.state_dict(),
                       f"{weight_dir}/best_model_fold_{fold}.pth")

            torch.save({
                "preds": all_preds,
                "labels": all_labels
            }, f"{result_dir}/predictions.pt")

    # ===== FINAL METRICS =====
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    metrics = {
        "accuracy": best_acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    # Save metrics
    with open(f"{result_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Save history
    with open(f"{result_dir}/history.json", "w") as f:
        json.dump(history, f, indent=4)

    print(f"✅ Fold {fold} Best Accuracy: {best_acc:.4f}")

    return metrics


# ===========================
# MAIN TRAINING LOOP
# ===========================
def main():

    all_metrics = []

    for fold in range(1, 6):
        print(f"\n===== Fold {fold} =====")

        model = ConvNeXtVisionMambaTabNet(num_classes=NUM_CLASSES)

        train_loader, val_loader = get_dataloaders_for_fold(
            fold=fold,
            batch_size=BATCH_SIZE,
            dataset_name=DATASET_NAME
        )

        metrics = train_one_fold(model, train_loader, val_loader, fold)
        all_metrics.append(metrics)

    # ===== SAVE OVERALL RESULTS =====
    overall_path = f"results/{DATASET_NAME}/overall_metrics.json"
    os.makedirs(f"results/{DATASET_NAME}", exist_ok=True)

    with open(overall_path, "w") as f:
        json.dump(all_metrics, f, indent=4)

    print("\n🎯 Training Complete for all folds!")


# ===========================
# RUN
# ===========================
if __name__ == "__main__":
    main()