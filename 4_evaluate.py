"""
4_evaluate.py - DAG Node 5: evaluate the trained model.

There is no labeled test split for this dataset (raw_data/test/ has raw
images with no class folders), so per the project spec we evaluate on
clean_data/valid/ - the same split used for validation during training -
and report that as the final test metric.
"""

import os
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from labels import get_name

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VALID_DIR = os.path.join(THIS_DIR, "clean_data", "valid")
MODEL_PATH = os.path.join(THIS_DIR, "model.pth")
CLASS_TO_IDX_PATH = os.path.join(THIS_DIR, "class_to_idx.json")
EVAL_RESULTS_PATH = os.path.join(THIS_DIR, "eval_results.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
TOP_K = 5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def build_model(num_classes):
    """Same architecture as 3_train.py - weights=None since we load our own checkpoint."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def main():
    print(f"FlowerLens AI - Step 4: Evaluating (device={DEVICE})")

    if not os.path.exists(MODEL_PATH):
        print("ERROR: model.pth not found. Run 3_train.py first.")
        return
    if not os.path.isdir(VALID_DIR):
        print("ERROR: clean_data/valid not found. Run 1_clean_data.py first.")
        return

    with open(CLASS_TO_IDX_PATH, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    test_ds = datasets.ImageFolder(VALID_DIR, transform=eval_transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model(num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    top1_correct = 0
    top5_correct = 0
    total = 0

    per_class_correct = {c: 0 for c in class_to_idx.values()}
    per_class_total = {c: 0 for c in class_to_idx.values()}

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, top5_preds = outputs.topk(TOP_K, dim=1)

            for i in range(labels.size(0)):
                true_label = labels[i].item()
                preds = top5_preds[i].tolist()

                per_class_total[true_label] += 1
                if preds[0] == true_label:
                    top1_correct += 1
                    per_class_correct[true_label] += 1
                if true_label in preds:
                    top5_correct += 1

                total += 1

    top1_acc = 100.0 * top1_correct / total
    top5_acc = 100.0 * top5_correct / total

    per_class_acc = {}
    for c in class_to_idx.values():
        folder_name = idx_to_class[c]
        denom = per_class_total[c]
        acc = 100.0 * per_class_correct[c] / denom if denom else 0.0
        per_class_acc[folder_name] = {
            "name": get_name(folder_name),
            "accuracy": acc,
            "total": denom,
            "correct": per_class_correct[c],
        }

    sorted_classes = sorted(per_class_acc.items(), key=lambda kv: kv[1]["accuracy"])
    worst5 = sorted_classes[:5]
    best5 = sorted_classes[-5:][::-1]

    print(f"\nTotal test images: {total}")
    print(f"Top-1 accuracy: {top1_acc:.1f}%")
    print(f"Top-5 accuracy: {top5_acc:.1f}%")

    print("\n5 best recognised classes:")
    for folder_name, info in best5:
        print(f"  {info['name']} ({folder_name}): {info['accuracy']:.1f}% ({info['correct']}/{info['total']})")

    print("\n5 worst recognised classes:")
    for folder_name, info in worst5:
        print(f"  {info['name']} ({folder_name}): {info['accuracy']:.1f}% ({info['correct']}/{info['total']})")

    results = {
        "total_test_images": total,
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "per_class_accuracy": per_class_acc,
        "best5": [{"class": n, **i} for n, i in best5],
        "worst5": [{"class": n, **i} for n, i in worst5],
    }

    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved evaluation results to {EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    main()
