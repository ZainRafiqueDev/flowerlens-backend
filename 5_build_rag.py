"""
5_build_rag.py - DAG Node 6: build the RAG (Retrieval-Augmented Generation)
reference store.

Strips the classifier head off the trained model so it acts as a pure
1280-dim feature extractor (the output of EfficientNet-B0's Global Average
Pooling), encodes up to MAX_PER_CLASS reference images per flower class from
clean_data/valid/, and saves everything needed for cosine-similarity lookup
at inference time into rag_store.pkl.

Paths are stored RELATIVE to backend/ (not absolute) so the same pickle
works unchanged after this whole folder is committed and deployed to Render.
"""

import os
import json
import pickle

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

from labels import get_name

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VALID_DIR = os.path.join(THIS_DIR, "clean_data", "valid")
MODEL_PATH = os.path.join(THIS_DIR, "model.pth")
CLASS_TO_IDX_PATH = os.path.join(THIS_DIR, "class_to_idx.json")
RAG_STORE_PATH = os.path.join(THIS_DIR, "rag_store.pkl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_PER_CLASS = 8

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

extract_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def build_feature_extractor(num_classes):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    # Strip the head AFTER loading weights - this turns the network into a
    # 1280-dim feature extractor (the output of Global Average Pooling).
    model.classifier = nn.Identity()
    model = model.to(DEVICE)
    model.eval()
    return model


def main():
    print(f"FlowerLens AI - Step 5: Building RAG store (device={DEVICE})")

    if not os.path.exists(MODEL_PATH):
        print("ERROR: model.pth not found. Run 3_train.py first.")
        return
    if not os.path.isdir(VALID_DIR):
        print("ERROR: clean_data/valid not found. Run 1_clean_data.py first.")
        return

    with open(CLASS_TO_IDX_PATH, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    num_classes = len(class_to_idx)

    extractor = build_feature_extractor(num_classes)

    class_folders = sorted(
        d for d in os.listdir(VALID_DIR) if os.path.isdir(os.path.join(VALID_DIR, d))
    )

    vectors = []
    labels = []
    names = []
    paths = []

    for class_name in class_folders:
        class_dir = os.path.join(VALID_DIR, class_name)
        files = sorted(f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f)))
        files = files[:MAX_PER_CLASS]

        display_name = get_name(class_name)

        for fname in files:
            fpath = os.path.join(class_dir, fname)
            try:
                img = Image.open(fpath).convert("RGB")
                tensor = extract_transform(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    vector = extractor(tensor).squeeze(0).cpu().numpy()

                vectors.append(vector)
                labels.append(class_name)
                names.append(display_name)
                # Stored RELATIVE to backend/ so rag.py can resolve it on any
                # machine (including Render, where absolute paths break).
                paths.append(os.path.relpath(fpath, THIS_DIR))
            except Exception as e:
                print(f"  [skip] {fpath}: {e}")

        print(f"  {display_name} ({class_name}): encoded {len(files)} images")

    vectors = np.stack(vectors, axis=0)

    store = {
        "vectors": vectors,
        "labels": labels,
        "paths": paths,
        "names": names,
    }

    with open(RAG_STORE_PATH, "wb") as f:
        pickle.dump(store, f)

    print(f"\nTotal reference vectors: {len(labels)} (dim={vectors.shape[1]})")
    print(f"Saved RAG store to {RAG_STORE_PATH}")


if __name__ == "__main__":
    main()
