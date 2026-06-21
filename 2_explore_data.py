"""
2_explore_data.py - DAG Node 2: dataset exploration.

Scans clean_data/train/, prints class distribution stats and average image
dimensions, and merges the results into dataset_summary.json.
"""

import os
import json
import random
from PIL import Image
from labels import get_name

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(THIS_DIR, "clean_data", "train")
SUMMARY_PATH = os.path.join(THIS_DIR, "dataset_summary.json")

SAMPLE_SIZE_FOR_DIMENSIONS = 200


def scan_classes(train_dir):
    class_folders = sorted(
        d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))
    )
    counts = {}
    all_files = []
    for class_name in class_folders:
        class_dir = os.path.join(train_dir, class_name)
        files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        counts[class_name] = len(files)
        all_files.extend(os.path.join(class_dir, f) for f in files)
    return counts, all_files


def sample_dimensions(file_paths, sample_size):
    sample = random.sample(file_paths, min(sample_size, len(file_paths)))
    widths, heights = [], []
    for path in sample:
        try:
            with Image.open(path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except Exception:
            continue
    if not widths:
        return 0, 0
    return sum(widths) / len(widths), sum(heights) / len(heights)


def main():
    print("FlowerLens AI - Step 2: Exploring clean_data/train/")

    if not os.path.isdir(TRAIN_DIR):
        print(f"ERROR: {TRAIN_DIR} not found. Run 1_clean_data.py first.")
        return

    counts, all_files = scan_classes(TRAIN_DIR)

    total_classes = len(counts)
    total_images = sum(counts.values())
    min_count = min(counts.values()) if counts else 0
    max_count = max(counts.values()) if counts else 0
    avg_count = total_images / total_classes if total_classes else 0

    sorted_by_count = sorted(counts.items(), key=lambda kv: kv[1])
    fewest = sorted_by_count[:5]
    most = sorted_by_count[-5:][::-1]

    avg_w, avg_h = sample_dimensions(all_files, SAMPLE_SIZE_FOR_DIMENSIONS)

    print(f"\nTotal classes : {total_classes}")
    print(f"Total images  : {total_images}")
    print(f"Min per class : {min_count}")
    print(f"Max per class : {max_count}")
    print(f"Avg per class : {avg_count:.1f}")

    print("\n5 classes with fewest images:")
    for name, count in fewest:
        print(f"  {get_name(name)} ({name}): {count}")

    print("\n5 classes with most images:")
    for name, count in most:
        print(f"  {get_name(name)} ({name}): {count}")

    print(f"\nSampled {min(SAMPLE_SIZE_FOR_DIMENSIONS, len(all_files))} images for dimensions")
    print(f"Average width : {avg_w:.1f}px")
    print(f"Average height: {avg_h:.1f}px")

    summary = {}
    if os.path.exists(SUMMARY_PATH):
        try:
            with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            summary = {}

    summary["exploration"] = {
        "total_classes": total_classes,
        "total_images": total_images,
        "min_per_class": min_count,
        "max_per_class": max_count,
        "avg_per_class": avg_count,
        "fewest_classes": [{"class": n, "name": get_name(n), "count": c} for n, c in fewest],
        "most_classes": [{"class": n, "name": get_name(n), "count": c} for n, c in most],
        "avg_width": avg_w,
        "avg_height": avg_h,
        "class_counts": counts,
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved exploration summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
