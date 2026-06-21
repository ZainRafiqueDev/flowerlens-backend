"""
main.py - FastAPI server for FlowerLens AI.

Loads every trained artifact once at startup, exposes a single POST /predict
endpoint that runs the 7-node DAG (see dag.py) end to end, and a GET / health
check. CORS is enabled for all origins so the React frontend (running on a
different port/domain) can call this freely.
"""

import os
import json
import pickle

import torch
import torch.nn as nn
from torchvision import models
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from dag import run_dag
from rag import load_extractor, search as rag_search
from labels import get_fact

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(THIS_DIR, "model.pth")
CLASS_TO_IDX_PATH = os.path.join(THIS_DIR, "class_to_idx.json")
CLASS_NAMES_PATH = os.path.join(THIS_DIR, "class_names.json")
RAG_STORE_PATH = os.path.join(THIS_DIR, "rag_store.pkl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="FlowerLens AI")

# CORSMiddleware must be added before any routes are defined.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- load artifacts once at startup -------------------------------------
with open(CLASS_TO_IDX_PATH, "r", encoding="utf-8") as f:
    class_to_idx = json.load(f)
idx_to_class = {v: k for k, v in class_to_idx.items()}
NUM_CLASSES = len(class_to_idx)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)


def _build_base_model():
    """Same architecture as 3_train.py - weights=None since we load our own checkpoint."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, NUM_CLASSES),
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    return model


# Full classifier (for softmax probabilities)
classifier_model = _build_base_model().to(DEVICE)
classifier_model.eval()

# Same weights, but with the head stripped off (for 1280-dim RAG vectors) -
# reuses rag.py's loader so there's exactly one place that builds an extractor.
extractor_model = load_extractor(MODEL_PATH)

with open(RAG_STORE_PATH, "rb") as f:
    rag_store = pickle.load(f)

print(f"FlowerLens AI backend ready - {NUM_CLASSES} classes, device={DEVICE}")


def model_fn(tensor):
    """tensor: normalized (1,3,224,224) torch.Tensor -> (probs, features)."""
    tensor = tensor.to(DEVICE)
    with torch.no_grad():
        logits = classifier_model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        features = extractor_model(tensor).squeeze(0).cpu().numpy()
    return probs, features


def rag_fn(features):
    return rag_search(features, rag_store, top_k=3)


@app.get("/")
def health():
    return {"status": "ok", "classes": NUM_CLASSES, "device": str(DEVICE)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw_bytes = await file.read()

    steps, top5, similar, is_confident = run_dag(raw_bytes, model_fn, rag_fn)

    top_class_id = top5[0]["class_id"]
    top_name = top5[0]["name"]
    top_confidence = top5[0]["confidence"]

    return {
        "prediction": top_name,
        "class_id": top_class_id,
        "confidence": top_confidence,
        "is_confident": is_confident,
        "top5": top5,
        "similar": similar,
        "dag_steps": steps,
        "fun_fact": get_fact(top_name),
    }


@app.get("/history")
def history():
    # Prediction history lives in MongoDB and is served by the Express
    # server (server/index.js -> GET /api/predictions), not FastAPI.
    return {
        "note": "Prediction history is stored and served by the Express server. "
                "See GET /api/predictions on the Node/Express backend."
    }
