import os
import csv
import random
import numpy as np
from PIL import Image
import json
from collections import Counter
from math import log

# ---------------- User-configurable params ----------------
messidor_dir = "Messidor_dataset"   # folder that contains the CSV and images (adjust)
messidor_csv_name = "messidor_data.csv" # csv filename inside messidor_dir
images_subdir = "messidor-2"                  # subfolder under messidor_dir where raw images live

out_images = "test.npy"
out_diag_onehot = "one_hot_labels.npy"
out_dme = "adjudicated_dme.npy"

img_size = (224, 224)   # resize shape (width, height)
filter_gradable = True  # whether to keep only rows with adjudicated_gradable == 1
random_seed = 1234
# --------------------------------------------------------

random.seed(random_seed)
np.random.seed(random_seed)

# Helper: robust image resolver
COMMON_EXTS = [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"]

def resolve_image_path(img_name: str, images_dir: str):
    """Robust resolution of Messidor image names."""
    candidate = os.path.join(images_dir, img_name)
    if os.path.exists(candidate):
        return candidate
    
    base, ext = os.path.splitext(img_name)
    if ext:
        for e in COMMON_EXTS:
            candidate = os.path.join(images_dir, base + e)
            if os.path.exists(candidate):
                return candidate
    
    for e in COMMON_EXTS:
        candidate = os.path.join(images_dir, img_name + e)
        if os.path.exists(candidate):
            return candidate
    
    csv_basename = os.path.splitext(img_name)[0].lower()
    try:
        for fname in os.listdir(images_dir):
            if os.path.splitext(fname)[0].lower() == csv_basename:
                return os.path.join(images_dir, fname)
    except FileNotFoundError:
        return None
    
    return None

# ------------------------- MAIN FUNCTION -----------------------------
def prepare_messidor(
    messidor_dir=messidor_dir,
    messidor_csv_name=messidor_csv_name,
    images_subdir=images_subdir,
    out_images=out_images,
    out_diag_onehot=out_diag_onehot,
    out_dme=out_dme,
    img_size=img_size,
    filter_gradable=filter_gradable
):
    csv_path = os.path.join(messidor_dir, messidor_csv_name)
    images_dir = os.path.join(messidor_dir, images_subdir)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images folder not found: {images_dir}")

    print("📂 Reading Messidor CSV (no pandas) ...")
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
            delimiter = dialect.delimiter
        except Exception:
            delimiter = ',' if ',' in sample else '\t'
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader)
        rows = [r for r in reader]

    if len(header) == 1 and (',' in header[0] or '\t' in header[0]):
        header = [h.strip() for h in header[0].split(delimiter)]

    header = [h.strip() for h in header]
    print(f"Detected delimiter: '{delimiter}'. Columns: {header}")

    # Find needed columns robustly
    def find_col(name_list):
        for c in name_list:
            if c in header:
                return header.index(c)
        raise RuntimeError(f"Missing required column among: {name_list}")

    idx_id = find_col(["id_code", "filename", "image"])
    idx_diag = find_col(["diagnosis", "label", "grade"])
    idx_dme = find_col(["adjudicated_dme", "dme"])
    
    gradable_idx = None
    for c in ["adjudicated_gradable", "gradable"]:
        if c in header:
            gradable_idx = header.index(c)

    # Build usable dataset entries
    records = []
    for row in rows:
        if gradable_idx is not None and filter_gradable:
            try:
                if int(float(row[gradable_idx])) != 1:
                    continue
            except:
                continue
        
        try:
            img_name = row[idx_id].strip()
            diagnosis = int(float(row[idx_diag]))
            dme = int(float(row[idx_dme]))
        except:
            continue
        
        records.append({"image": img_name, "diagnosis": diagnosis, "dme": dme})

    print(f"Filtered usable samples: {len(records)}")

    # -------------- Load images -----------------
    images = []
    diag_labels = []
    dme_labels = []
    missing = []

    for i, rec in enumerate(records):
        resolved = resolve_image_path(rec["image"], images_dir)
        if resolved is None:
            missing.append(rec["image"])
            continue

        try:
            with Image.open(resolved) as im:
                im = im.convert("RGB")
                im = im.resize(img_size)
                arr = np.array(im, dtype=np.uint8)
        except Exception as e:
            print(f"Error loading {resolved}: {e}")
            missing.append(rec["image"])
            continue

        # Diagnosis one-hot (5 classes)
        one_hot = np.zeros(5, dtype=np.uint8)
        if 0 <= rec["diagnosis"] <= 4:
            one_hot[rec["diagnosis"]] = 1
        else:
            continue

        images.append(arr)
        diag_labels.append(one_hot)
        dme_labels.append(rec["dme"])

        if (i + 1) % 200 == 0:
            print(f"Processed {i + 1}/{len(records)}")

    # Convert to numpy arrays
    images = np.array(images, dtype=np.uint8)
    diag_labels = np.array(diag_labels, dtype=np.uint8)
    dme_labels = np.array(dme_labels, dtype=np.uint8)

    print(f"\nFinal shapes:")
    print(f"  images: {images.shape}")
    print(f"  labels(one-hot): {diag_labels.shape}")
    print(f"  dme: {dme_labels.shape}")
    print(f"Missing entries: {len(missing)}")

    # Save arrays
    np.save(out_images, images)
    np.save(out_diag_onehot, diag_labels)
    np.save(out_dme, dme_labels)

    print("\n💾 Saved:")
    print(f"  {out_images}")
    print(f"  {out_diag_onehot}")
    print(f"  {out_dme}")

    # ------------------------------------------------------------------
    # ⭐ NEW: Compute global test metadata (used in GNN/FedMRL++)
    # ------------------------------------------------------------------
    label_counts = Counter(np.argmax(diag_labels, axis=1))
    total = sum(label_counts.values())

    # entropy (natural log)
    p = np.array([label_counts.get(i, 0) / total for i in range(5)], dtype=float)
    p_nonzero = p[p > 0]
    entropy = float(-np.sum(p_nonzero * np.log(p_nonzero)))

    metadata = {
        "num_samples": int(total),
        "class_distribution": {int(k): int(v) for k, v in label_counts.items()},
        "entropy": entropy,
        "missing_count": len(missing),
        "img_size": img_size
    }

# -----------------------------------------------------------------

if __name__ == "__main__":
    prepare_messidor()
