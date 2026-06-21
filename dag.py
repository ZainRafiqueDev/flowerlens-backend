"""
dag.py - the 7-node Directed Acyclic Graph that turns an uploaded image into
a prediction.

Each node does exactly one job and hands its output forward to the next
node. Data only ever flows forward - there are no cycles, no node ever
revisits an earlier step. That's precisely what makes this a DAG: a graph of
operations with a defined start and end and no loops, which is how every ML
inference pipeline is structured under the hood.
"""

import io
import time

import numpy as np
from PIL import Image
from torchvision import transforms

from labels import get_name

CONFIDENCE_THRESHOLD = 40.0  # percent - below this we flag LOW_CONFIDENCE

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_resize = transforms.Resize((224, 224))
_to_tensor_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def run_dag(image_bytes: bytes, model_fn, rag_fn):
    """
    Runs the 7-node inference DAG on raw uploaded image bytes.

    model_fn(tensor) -> (probs, features)
        tensor: normalized (1,3,224,224) torch.Tensor
        probs: np.ndarray of shape (num_classes,) - softmax probabilities
        features: np.ndarray of shape (1280,) - GAP latent feature vector

    rag_fn(features) -> list of {name, class_id, similarity, image_b64}
        cosine-similarity lookup against the pre-built reference store

    Returns: (steps, top5, similar, is_confident)
    """
    start = time.time()
    steps = []

    def log(node, detail):
        steps.append({
            "node": node,
            "detail": detail,
            "ms": int((time.time() - start) * 1000),
        })

    # ---- Node 1: Receive & Decode -----------------------------------
    raw_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = raw_image.size
    log(
        "Receive & Decode",
        f"Decoded the uploaded bytes into a PIL RGB image. Original size: {orig_w}x{orig_h}px.",
    )

    # ---- Node 2: Resize to 224x224 ------------------------------------
    resized = _resize(raw_image)
    log(
        "Resize to 224x224",
        "Resized to 224x224px because EfficientNet-B0's pretrained convolution "
        "filters and the trained classifier head expect this fixed input resolution.",
    )

    # ---- Node 3: Normalise Pixels --------------------------------------
    tensor = _to_tensor_normalize(resized).unsqueeze(0)
    log(
        "Normalise Pixels",
        f"Scaled pixels to [0,1] then subtracted the ImageNet mean {IMAGENET_MEAN} and "
        f"divided by std {IMAGENET_STD}, so this image's pixel statistics match what "
        "the pretrained network saw during ImageNet training.",
    )

    # ---- Node 4: CNN Feature Extraction ---------------------------------
    probs, features = model_fn(tensor)
    log(
        "CNN Feature Extraction",
        f"EfficientNet-B0's depthwise convolutions + Global Average Pooling reduced "
        f"the image to a {features.shape[0]}-dim latent feature vector.",
    )

    # ---- Node 5: Softmax Classification ----------------------------------
    top5_idx = np.argsort(probs)[::-1][:5]
    top5 = [
        {
            "class_id": int(i),
            "name": get_name(int(i)),
            "confidence": float(probs[i] * 100),
        }
        for i in top5_idx
    ]
    top1_conf = top5[0]["confidence"]
    log(
        "Softmax Classification",
        f"The linear classifier head produced {len(probs)} class scores; softmax "
        f"converted them to probabilities. Top prediction: '{top5[0]['name']}' at {top1_conf:.1f}%.",
    )

    # ---- Node 6: Confidence Threshold Check -------------------------------
    is_confident = top1_conf >= CONFIDENCE_THRESHOLD
    log(
        "Confidence Threshold Check",
        f"Top confidence {top1_conf:.1f}% vs the {CONFIDENCE_THRESHOLD:.0f}% threshold -> "
        f"{'CONFIDENT' if is_confident else 'LOW_CONFIDENCE'}.",
    )

    # ---- Node 7: RAG Cosine Retrieval --------------------------------------
    similar = rag_fn(features)
    similar_names = ", ".join(s["name"] for s in similar) if similar else "none"
    log(
        "RAG Cosine Retrieval",
        f"Compared the {features.shape[0]}-dim feature vector against the reference "
        f"store via cosine similarity and retrieved the top {len(similar)} matches: {similar_names}.",
    )

    return steps, top5, similar, is_confident
