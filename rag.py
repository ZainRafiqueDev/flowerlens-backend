"""
rag.py - feature extraction + cosine similarity search for RAG retrieval.

Mirrors the EfficientNet-B0 feature extractor built in 5_build_rag.py, so a
freshly uploaded image and the pre-computed rag_store.pkl vectors live in
the exact same 1280-dim latent space and are directly comparable.
"""

import os
import json
import base64

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_extractor(model_path):
    """Load the trained EfficientNet-B0 and strip its classifier head so it
    returns a 1280-dim feature vector (Global Average Pooling output)."""
    class_to_idx_path = os.path.join(THIS_DIR, "class_to_idx.json")
    with open(class_to_idx_path, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    num_classes = len(class_to_idx)

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    model.classifier = nn.Identity()
    model = model.to(DEVICE)
    model.eval()
    return model


def get_vector(extractor, img):
    """img: PIL.Image (RGB or convertible). Returns a 1280-dim numpy vector."""
    tensor = _transform(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        vector = extractor(tensor).squeeze(0).cpu().numpy()
    return vector


def _image_to_b64(relative_path):
    """rag_store.pkl stores paths RELATIVE to backend/ - resolve them here
    so this works the same on a local machine and on Render."""
    full_path = os.path.join(THIS_DIR, relative_path)
    try:
        with open(full_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def search(query_vec, store, top_k=3):
    """Cosine similarity between query_vec and every vector in store.
    Returns top_k dicts: {name, class_id, similarity, image_b64}."""
    vectors = store["vectors"]  # (N, 1280)
    labels = store["labels"]
    names = store["names"]
    paths = store["paths"]

    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    store_norms = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)

    similarities = store_norms @ query_norm  # (N,) cosine similarity, range [-1, 1]

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        image_b64 = _image_to_b64(paths[idx])
        results.append({
            "name": names[idx],
            "class_id": labels[idx],
            "similarity": float(similarities[idx] * 100),
            "image_b64": image_b64,
        })

    return results
