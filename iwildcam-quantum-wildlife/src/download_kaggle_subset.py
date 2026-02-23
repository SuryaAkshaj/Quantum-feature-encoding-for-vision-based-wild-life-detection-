"""
download_kaggle_subset.py
--------------------------
Downloads a small real subset of iWildCam 2020.

TWO MODES (auto-detected):
 1. Manual mode (recommended): Place files you downloaded from the browser
    into data/iwildcam_v2.0/kaggle_tmp/ and run this script.
 2. API mode: Uses kaggle CLI if credentials work.

Manual download steps (browser):
 a) Go to: https://www.kaggle.com/c/iwildcam-2020-fgvc7/data
 b) Accept competition rules if prompted
 c) Download: train.json   (~20 MB)  → place in data/iwildcam_v2.0/
 d) Download: train.zip    (~39 GB)  → OR download a small sample below
    For a small sample, click individual image files or use the
    "Download (All)" button for just the metadata files.

Alternatively, download train.json directly from:
 https://storage.googleapis.com/public-datasets-lila/iwildcam/2022/iwildcam2022_mdv4_detections_train.json

After placing train.json:
    python -m src.download_kaggle_subset

Usage:
    python -m src.download_kaggle_subset
"""

import os, json, shutil, zipfile, collections, csv
from pathlib import Path

# ── Settings ──────────────────────────────────────────────────────────────────
COMPETITION    = "iwildcam-2020-fgvc7"
DATA_DIR       = Path("data/iwildcam_v2.0")
IMG_DIR        = DATA_DIR / "extracted" / "train"
META_JSON      = DATA_DIR / "train.json"
DOWNLOAD_TMP   = DATA_DIR / "kaggle_tmp"
MAX_IMAGES     = 3000
TOP_N_SPECIES  = 20

os.makedirs(IMG_DIR,      exist_ok=True)
os.makedirs(DOWNLOAD_TMP, exist_ok=True)


# ── Step 1: Get train.json ────────────────────────────────────────────────────
def get_metadata():
    if META_JSON.exists() and META_JSON.stat().st_size > 500_000:
        print(f"train.json ready ({META_JSON.stat().st_size // 1024} KB)")
        return True

    # Try Kaggle API
    print("Attempting Kaggle API download...")
    ret = os.system(
        f'kaggle competitions download -c {COMPETITION} -f train.json -p "{DOWNLOAD_TMP}"'
    )
    if ret == 0:
        for candidate in [DOWNLOAD_TMP / "train.json", DOWNLOAD_TMP / "train.json.zip"]:
            if candidate.exists():
                if candidate.suffix == ".zip":
                    with zipfile.ZipFile(candidate) as z:
                        z.extractall(DOWNLOAD_TMP)
                    candidate.unlink()
                src = DOWNLOAD_TMP / "train.json"
                if src.exists():
                    shutil.move(str(src), str(META_JSON))
                    print(f"train.json saved ({META_JSON.stat().st_size // 1024} KB)")
                    return True

    # Manual fallback instructions
    print("\n" + "=" * 60)
    print("KAGGLE API FAILED — Manual download needed (2 minutes):")
    print("=" * 60)
    print()
    print("1. Open this URL in your browser:")
    print("   https://www.kaggle.com/c/iwildcam-2020-fgvc7/data")
    print()
    print("2. Accept the competition rules if prompted")
    print()
    print("3. Download 'train.json' from the file list")
    print("   (Click the file name → Download button)")
    print()
    print(f"4. Place it here:")
    print(f"   {META_JSON.resolve()}")
    print()
    print("5. Run this script again:")
    print("   python -m src.download_kaggle_subset")
    print("=" * 60)
    return False


# ── Step 2: Parse top species ─────────────────────────────────────────────────
def pick_images():
    print("\nParsing metadata...")
    with open(META_JSON) as f:
        data = json.load(f)

    id_to_file  = {img["id"]: os.path.basename(img["file_name"])
                   for img in data.get("images", [])}
    id_to_cat   = {ann["image_id"]: ann["category_id"]
                   for ann in data.get("annotations", [])}
    cat_to_name = {c["id"]: c["name"] for c in data.get("categories", [])}

    cat_counts  = collections.Counter(id_to_cat.values())
    top_cats    = [cat for cat, _ in cat_counts.most_common(TOP_N_SPECIES)]

    print(f"\nTop {TOP_N_SPECIES} species in dataset:")
    for cat in top_cats:
        print(f"  {cat_to_name.get(cat, cat):30s}  {cat_counts[cat]:6d} images")

    per_class = MAX_IMAGES // TOP_N_SPECIES
    selected  = []
    cat_seen  = collections.defaultdict(int)

    for img_id, cat_id in id_to_cat.items():
        if cat_id not in top_cats:
            continue
        if cat_seen[cat_id] >= per_class:
            continue
        fname = id_to_file.get(img_id, "")
        if not fname:
            continue
        selected.append((img_id, fname, cat_id))
        cat_seen[cat_id] += 1

    print(f"\nSelected {len(selected)} images across "
          f"{len(set(c for _, _, c in selected))} species")
    return selected, cat_to_name


# ── Step 3: Download or check images ─────────────────────────────────────────
def get_images(selected):
    existing = {f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")}
    needed   = [row for row in selected if row[1] not in existing]
    print(f"\nAlready have {len(existing)} images. Need {len(needed)} more.")

    if not needed:
        return

    # Try Kaggle API for full zip
    print("Attempting Kaggle API image download (~39 GB zip — may be slow)...")
    ret = os.system(
        f'kaggle competitions download -c {COMPETITION} -f train.zip -p "{DOWNLOAD_TMP}"'
    )

    if ret != 0:
        needed_names = {row[1] for row in needed}
        print("\n" + "=" * 60)
        print("MANUAL IMAGES — Place them in:")
        print(f"   {IMG_DIR.resolve()}")
        print()
        print("Option 1: Kaggle competition page → Data tab")
        print("  https://www.kaggle.com/c/iwildcam-2020-fgvc7/data")
        print("  Download train.zip (39 GB) — only if you have space/time")
        print()
        print("Option 2: Use only images you already have")
        print("  The script will skip missing images automatically.")
        print("=" * 60)
        print("\nContinuing with existing images only...")
        return

    # Extract selectively from zip
    zip_path = DOWNLOAD_TMP / "train.zip"
    if zip_path.exists():
        needed_names = {row[1] for row in needed}
        print(f"Extracting {len(needed_names)} images from zip...")
        with zipfile.ZipFile(zip_path) as z:
            for member in z.namelist():
                fname = os.path.basename(member)
                if fname in needed_names:
                    with z.open(member) as src, open(IMG_DIR / fname, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        zip_path.unlink()


# ── Step 4: Save labels CSV ───────────────────────────────────────────────────
def save_labels(selected, cat_to_name):
    # Match to files that actually exist on disk
    existing = {f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")}
    matched  = [(img_id, fname, cat_id) for img_id, fname, cat_id in selected
                if fname in existing]

    if not matched:
        print("\nNo images matched! Exiting.")
        return

    unique_cats = sorted(set(c for _, _, c in matched))
    cat_to_idx  = {cat: idx for idx, cat in enumerate(unique_cats)}
    dense_names = {str(idx): cat_to_name.get(cat, f"species_{cat}")
                   for cat, idx in cat_to_idx.items()}

    csv_path = DATA_DIR / "image_labels.csv"
    count = 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label"])
        for _, fname, cat_id in matched:
            writer.writerow([str(IMG_DIR / fname), cat_to_idx[cat_id]])
            count += 1

    os.makedirs("models", exist_ok=True)
    with open("models/class_names.json", "w") as f:
        json.dump(dense_names, f, indent=2)

    print(f"\nSaved {count} images to {csv_path}")
    print(f"Classes ({len(dense_names)}):")
    for idx, name in sorted(dense_names.items(), key=lambda x: int(x[0])):
        c = sum(1 for _, _, ci in matched if cat_to_idx[ci] == int(idx))
        print(f"  [{idx:>2}] {name:30s}  {c} images")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not get_metadata():
        raise SystemExit(0)

    selected, cat_to_name = pick_images()
    get_images(selected)
    save_labels(selected, cat_to_name)

    print("\n--- Done! Next steps ---")
    print("  python -m src.precompute_local")
    print("  python -m src.train_mlp_local")
