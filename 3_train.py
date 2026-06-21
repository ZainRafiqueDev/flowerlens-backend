"""
3_train.py - DAG Nodes 3+4: data augmentation + EfficientNet-B0 training.

Trains on clean_data/train/, validates on clean_data/valid/ after every
epoch, and checkpoints model.pth whenever validation accuracy improves.

Phase 1 (epochs 1..UNFREEZE_EPOCH-1): EfficientNet-B0's pretrained feature
extractor is frozen; only the new classifier head is trained.
Phase 2 (epoch UNFREEZE_EPOCH onward): the whole network is unfrozen and
fine-tuned at a 10x lower learning rate.
"""

import os
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(THIS_DIR, "clean_data", "train")
VALID_DIR = os.path.join(THIS_DIR, "clean_data", "valid")

MODEL_PATH = os.path.join(THIS_DIR, "model.pth")
CLASS_TO_IDX_PATH = os.path.join(THIS_DIR, "class_to_idx.json")
IDX_TO_CLASS_PATH = os.path.join(THIS_DIR, "idx_to_class.json")
CLASS_NAMES_PATH = os.path.join(THIS_DIR, "class_names.json")
HISTORY_PATH = os.path.join(THIS_DIR, "training_history.json")

# ---- config -------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
UNFREEZE_EPOCH = 6  # 1-indexed epoch at which we unfreeze every layer
NUM_WORKERS = 2     # bump up if your CPU/disk can keep the GPU fed; 0 is safest

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def build_model(num_classes):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Freeze the pretrained feature extractor for phase 1.
    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features  # 1280 for EfficientNet-B0
    new_classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    nn.init.kaiming_normal_(new_classifier[1].weight, nonlinearity="relu")
    nn.init.zeros_(new_classifier[1].bias)
    model.classifier = new_classifier

    return model.to(DEVICE)


def run_epoch(model, loader, criterion, optimizer=None):
    """One pass over loader. Trains if optimizer is given, else just evaluates."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            if is_train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def main():
    print(f"FlowerLens AI - Step 3: Training (device={DEVICE})")

    if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(VALID_DIR):
        print("ERROR: clean_data/train or clean_data/valid not found. Run 1_clean_data.py first.")
        return

    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    valid_ds = datasets.ImageFolder(VALID_DIR, transform=valid_transform)

    # NUM_CLASSES is detected at runtime - never hardcode it (folder count varies).
    NUM_CLASSES = len(train_ds.classes)
    print(f"Detected {NUM_CLASSES} classes from clean_data/train/")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # All three label files are derived from the SAME train_ds.class_to_idx,
    # so the model's output index always lines up with the right class name.
    class_to_idx = train_ds.class_to_idx        # e.g. {"1": 0, "10": 1, ...}
    idx_to_class = {str(v): k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[str(i)] for i in range(NUM_CLASSES)]

    with open(CLASS_TO_IDX_PATH, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)
    with open(IDX_TO_CLASS_PATH, "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, indent=2)
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    model = build_model(NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()

    # Phase 1: only the new classifier head is trainable.
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    history = []
    unfrozen = False

    for epoch in range(1, EPOCHS + 1):
        if epoch == UNFREEZE_EPOCH and not unfrozen:
            new_lr = LEARNING_RATE * 0.1
            print(f"\n>> Epoch {epoch}: unfreezing all layers, reducing LR to {new_lr}\n")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=new_lr, weight_decay=WEIGHT_DECAY)
            remaining_epochs = EPOCHS - epoch + 1
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)
            unfrozen = True

        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, valid_loader, criterion, optimizer=None)
        scheduler.step()
        elapsed = time.time() - start

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss:{train_loss:.3f} Acc:{train_acc:.1f}% | "
            f"Val Loss:{val_loss:.3f} Acc:{val_acc:.1f}% | "
            f"{elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> new best val_acc={val_acc:.1f}%, saved model.pth")

    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val_acc={best_val_acc:.1f}%")
    print(f"Saved: {MODEL_PATH}")
    print(f"Saved: {CLASS_TO_IDX_PATH}, {IDX_TO_CLASS_PATH}, {CLASS_NAMES_PATH}, {HISTORY_PATH}")


if __name__ == "__main__":
    main()
