"""
1_clean_data.py - DAG Node 1: data cleaning.

Scans raw_data/train/ and raw_data/valid/ (the Oxford 102 dataset folders),
drops any image that is corrupt or smaller than 32x32px, and copies the
survivors into clean_data/train/ and clean_data/valid/ preserving the class
folder structure. raw_data/test/ is never touched - it has no labels.
"""

import os
import json
from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# The Oxford 102 dataset lives in oxford/flower_data at the project root,
# two levels above backend/. Override with the FLOWERLENS_RAW_DIR env var
# if you move the dataset somewhere else.
RAW_DIR = os.environ.get(
    "FLOWERLENS_RAW_DIR",
    os.path.join(THIS_DIR, "..", "..", "oxford", "flower_data"),
)
CLEAN_DIR = os.path.join(THIS_DIR, "clean_data")
SUMMARY_PATH = os.path.join(THIS_DIR, "dataset_summary.json")

MIN_SIZE = 32
SPLITS = ["train", "valid"]  # never touch test/ - it has no labels


def is_image_ok(path):
    """Returns (ok, reason). Verifies the file isn't corrupt and is >= 32x32."""
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception:
        return False, "corrupt"

    try:
        with Image.open(path) as img:
            width, height = img.size
            if width < MIN_SIZE or height < MIN_SIZE:
                return False, "too_small"
    except Exception:
        return False, "corrupt"

    return True, None


def clean_split(split):
    src_root = os.path.join(RAW_DIR, split)
    dst_root = os.path.join(CLEAN_DIR, split)

    if not os.path.isdir(src_root):
        print(f"  [skip] {src_root} does not exist")
        return {"kept": 0, "corrupt": 0, "too_small": 0, "classes": 0}

    os.makedirs(dst_root, exist_ok=True)

    class_folders = sorted(
        d for d in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, d))
    )

    kept, corrupt, too_small = 0, 0, 0

    for class_name in class_folders:
        src_class_dir = os.path.join(src_root, class_name)
        dst_class_dir = os.path.join(dst_root, class_name)
        os.makedirs(dst_class_dir, exist_ok=True)

        for fname in os.listdir(src_class_dir):
            src_path = os.path.join(src_class_dir, fname)
            if not os.path.isfile(src_path):
                continue

            ok, reason = is_image_ok(src_path)
            if not ok:
                if reason == "corrupt":
                    corrupt += 1
                else:
                    too_small += 1
                continue

            try:
                with Image.open(src_path) as img:
                    rgb = img.convert("RGB")
                    dst_path = os.path.join(dst_class_dir, fname)
                    rgb.save(dst_path)
                kept += 1
            except Exception:
                corrupt += 1

    print(f"  {split}: kept={kept} corrupt={corrupt} too_small={too_small} classes={len(class_folders)}")
    return {"kept": kept, "corrupt": corrupt, "too_small": too_small, "classes": len(class_folders)}


def main():
    print("FlowerLens AI - Step 1: Cleaning data")
    print(f"Raw data dir  : {os.path.abspath(RAW_DIR)}")
    print(f"Clean data dir: {os.path.abspath(CLEAN_DIR)}")
    print("NOTE: raw_data/test/ is intentionally skipped - it has no labels.\n")

    report = {}
    for split in SPLITS:
        print(f"Cleaning '{split}'...")
        report[split] = clean_split(split)

    total_kept = sum(r["kept"] for r in report.values())
    total_corrupt = sum(r["corrupt"] for r in report.values())
    total_too_small = sum(r["too_small"] for r in report.values())

    print("\n=== Cleaning summary ===")
    print(f"Total kept      : {total_kept}")
    print(f"Total corrupt   : {total_corrupt}")
    print(f"Total too small : {total_too_small}")

    summary = {}
    if os.path.exists(SUMMARY_PATH):
        try:
            with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            summary = {}

    summary["cleaning"] = {
        "raw_dir": os.path.abspath(RAW_DIR),
        "splits": report,
        "total_kept": total_kept,
        "total_corrupt": total_corrupt,
        "total_too_small": total_too_small,
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved cleaning report to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
