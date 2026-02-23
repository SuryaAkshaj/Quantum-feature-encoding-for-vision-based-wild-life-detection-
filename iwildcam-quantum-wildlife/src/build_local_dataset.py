"""
build_local_dataset.py
-----------------------
Downloads the iWildCam 2020 train.json metadata (annotations only, no images),
maps the 670 extracted local image filenames -> category_id (species label),
remaps to 0-indexed labels, and saves:
  - data/iwildcam_v2.0/image_labels.csv
  - models/class_names.json

Usage:
    python -m src.build_local_dataset
"""

import os
import json
import csv
import urllib.request
import collections

TRAIN_JSON_URL = (
    "https://raw.githubusercontent.com/visipedia/iwildcam_comp/"
    "master/previous_years/2020/data/train.json"
)
# Fallback: WILDS-hosted metadata
WILDS_METADATA_URL = (
    "https://worksheets.codalab.org/rest/bundles/0xed677e00edb94fa2b77ef6cce47742b9/"
    "contents/blob/metadata"
)

DATA_DIR   = os.path.join("data", "iwildcam_v2.0")
IMG_DIR    = os.path.join(DATA_DIR, "extracted", "train")
OUT_CSV    = os.path.join(DATA_DIR, "image_labels.csv")
META_JSON  = os.path.join(DATA_DIR, "train.json")
CLASS_JSON = os.path.join("models", "class_names.json")

# ────────────────────────────────────────────────────────────────────────────
def download_metadata():
    """Download train.json if not already present."""
    if os.path.exists(META_JSON):
        print(f"✔ Metadata already present: {META_JSON}")
        return True

    print("⬇  Downloading iWildCam 2020 train.json metadata …")
    for url in [TRAIN_JSON_URL, WILDS_METADATA_URL]:
        try:
            urllib.request.urlretrieve(url, META_JSON)
            size = os.path.getsize(META_JSON) / 1024
            print(f"   ✔ Downloaded {size:.0f} KB from {url}")
            return True
        except Exception as e:
            print(f"   ✗ Failed ({url}): {e}")
    return False


def build_dataset_from_json():
    """Parse train.json and match to local images."""
    with open(META_JSON, "r") as f:
        data = json.load(f)

    # Build image_id -> filename mapping
    id_to_file = {}
    for img in data.get("images", []):
        fname = os.path.basename(img["file_name"])
        id_to_file[img["id"]] = fname

    # Build image_id -> category_id mapping
    id_to_cat = {}
    for ann in data.get("annotations", []):
        id_to_cat[ann["image_id"]] = ann["category_id"]

    # Build category_id -> name mapping
    cat_id_to_name = {}
    for cat in data.get("categories", []):
        cat_id_to_name[cat["id"]] = cat["name"]

    # Get local image filenames (uuid format)
    local_files = set(os.listdir(IMG_DIR))
    print(f"\nLocal images: {len(local_files)}")
    print(f"Annotations:  {len(id_to_cat)}")

    # Match local images to labels
    matched = []  # (filepath, category_id, category_name)
    for img_id, fname in id_to_file.items():
        if fname in local_files and img_id in id_to_cat:
            cat_id = id_to_cat[img_id]
            cat_name = cat_id_to_name.get(cat_id, f"species_{cat_id}")
            matched.append((os.path.join(IMG_DIR, fname), cat_id, cat_name))

    print(f"Matched:      {len(matched)} images with labels")
    return matched, cat_id_to_name


def build_dataset_without_json():
    """
    Fallback: no JSON metadata available.
    Use YOLO to detect objects and group images into:
      - background (no detections)
      - animal_N  (by YOLO class index, e.g. 'cat', 'dog', 'bird', etc.)
    This gives real (vision-based) labels even without iWildCam annotations.
    """
    print("\n⚠  No train.json found. Using YOLO-based pseudo-labeling …")
    from ultralytics import YOLO

    yolo = YOLO("yolov8n.pt")
    local_files = sorted(os.listdir(IMG_DIR))

    # COCO class names from YOLOv8 (indices 14-23 are animals)
    ANIMAL_CLASSES = {
        14: "bird", 15: "cat", 16: "dog", 17: "horse",
        18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
        22: "zebra", 23: "giraffe",
    }

    label_counter = collections.Counter()
    rows = []

    print(f"Running YOLO on {len(local_files)} images …")
    for i, fname in enumerate(local_files):
        if not fname.lower().endswith(".jpg"):
            continue
        fpath = os.path.join(IMG_DIR, fname)
        try:
            results = yolo.predict(source=fpath, conf=0.25, device="cpu",
                                   verbose=False, show=False)
            detected_classes = []
            for r in results:
                if r.boxes is not None:
                    for cls_t in r.boxes.cls.cpu().numpy():
                        detected_classes.append(int(cls_t))

            # Pick best animal class, else "background"
            animal_hits = [c for c in detected_classes if c in ANIMAL_CLASSES]
            if animal_hits:
                top_cls = collections.Counter(animal_hits).most_common(1)[0][0]
                label_name = ANIMAL_CLASSES[top_cls]
            else:
                label_name = "background"

            rows.append((fpath, label_name))
            label_counter[label_name] += 1

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(local_files)}] {label_counter}")
        except Exception as e:
            print(f"  skip {fname}: {e}")

    print(f"\nLabel distribution: {dict(label_counter)}")

    # Remap to integer labels
    unique_labels = sorted(label_counter.keys())
    label_to_id   = {l: i for i, l in enumerate(unique_labels)}
    matched = [(fp, label_to_id[lbl], lbl) for fp, lbl in rows]
    cat_id_to_name = {v: k for k, v in label_to_id.items()}
    return matched, cat_id_to_name


# ────────────────────────────────────────────────────────────────────────────
def remap_labels(matched):
    """Remap sparse category_ids to dense 0-indexed integers."""
    raw_cats = sorted(set(c for _, c, _ in matched))
    remap = {old: new for new, old in enumerate(raw_cats)}
    remapped = [(fp, remap[cat_id], name) for fp, cat_id, name in matched]
    # Build dense label map
    dense_name_map = {}  # int label -> name
    for fp, new_id, name in remapped:
        dense_name_map[new_id] = name
    return remapped, dense_name_map


def save_csv(matched, out_path):
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label"])
        for fp, label_id, _ in matched:
            writer.writerow([fp, label_id])
    print(f"✔ Saved CSV: {out_path}  ({len(matched)} rows)")


def save_class_names(dense_name_map, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Convert int keys to str for JSON
    payload = {str(k): v for k, v in sorted(dense_name_map.items())}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"✔ Saved class names: {out_path}  ({len(payload)} classes)")
    for k, v in sorted(payload.items(), key=lambda x: int(x[0])):
        print(f"   [{k}] {v}")


# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Step 1: try to download+use train.json
    got_json = download_metadata()

    if got_json and os.path.exists(META_JSON):
        try:
            matched, _ = build_dataset_from_json()
        except Exception as e:
            print(f"train.json parse error: {e}")
            matched = []
    else:
        matched = []

    # Step 2: if no JSON matches, fall back to YOLO pseudo-labels
    if len(matched) < 10:
        print(f"\nOnly {len(matched)} JSON matches — using YOLO pseudo-labeling instead.")
        matched, raw_name_map = build_dataset_without_json()
        dense_name_map = {v: k for k, v in {v: k for k, v in raw_name_map.items()}.items()}
        # matched already remapped in build_dataset_without_json
    else:
        matched, dense_name_map = remap_labels(matched)

    if len(matched) == 0:
        print("❌ No labeled images found. Cannot proceed.")
        exit(1)

    # Step 3: show class distribution
    class_counts = collections.Counter(lbl for _, lbl, _ in matched)
    print(f"\nClass distribution ({len(class_counts)} classes, {len(matched)} samples):")
    for cls_id, cnt in sorted(class_counts.items()):
        name = dense_name_map.get(cls_id, f"class_{cls_id}")
        print(f"   [{cls_id:3d}] {name:30s}  {cnt:4d} images")

    # Step 4: save outputs
    save_csv(matched, OUT_CSV)
    save_class_names(dense_name_map, CLASS_JSON)

    print("\n✅ Dataset build complete!")
    print(f"   Next: python -m src.precompute_local")
