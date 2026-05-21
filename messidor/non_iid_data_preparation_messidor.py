import os
import csv
import random
import shutil
import numpy as np

# ------------------ User parameters (edit as needed) ------------------
messidor_csv = "Messidor_dataset/messidor_data.csv"   # path to the Messidor CSV/TSV
messidor_images = "Messidor_dataset/messidor-2"  # folder with images
output_folder = "messidor_alpha_1.0"
os.makedirs(output_folder, exist_ok=True)

# Non-IID partitioning params
num_clients = 4
num_shards_per_class = 200
eta = 1.0
num_samples_per_shard = 5

# Train/test split per client (80% train, 20% test)
train_frac = 0.8

# Optional server test set
create_server_test = True
server_test_count = 240

# Reproducibility
random_seed = 1234
random.seed(random_seed)
np.random.seed(random_seed)
# ----------------------------------------------------------------------

# ----- helper: resolve image path handling many extensions/case -----
COMMON_EXTS = [".png", ".PNG", ".jpg", ".JPG", ".jpeg", ".JPEG"]

def resolve_image_path(img_name: str, images_dir: str):
    """
    Resolve the actual image filepath for a given CSV entry img_name.
    Tries common extensions and case variations.
    Returns the full path if found, otherwise None.
    """
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
# ------------------------------------------------------------------------------

# ---------------- Robust CSV read (no pandas) ----------------

with open(messidor_csv, "r", encoding="utf-8-sig") as f:
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
    if ',' in header[0]:
        header = [h.strip() for h in header[0].split(',')]
    else:
        header = [h.strip() for h in header[0].split('\t')]

print(f"✅ CSV read successfully with {len(rows)} rows. Detected delimiter: '{delimiter}'")
# -----------------------------------------------------------------

# normalize header and find indexes robustly
header = [h.strip() for h in header]
try:
    idx_id = header.index("id_code")
    idx_diag = header.index("diagnosis")
    idx_gradable = header.index("adjudicated_gradable")
except ValueError as e:
    raise RuntimeError(f"Expected column not found in header: {e}. Header: {header}")

# Build data (filter gradable==1)
data = []
for row in rows:
    if len(row) <= max(idx_id, idx_diag, idx_gradable):
        continue
    try:
        gradable = int(float(row[idx_gradable]))
    except:
        continue
    if gradable != 1:
        continue
    img_name = row[idx_id].strip()
    try:
        diagnosis = int(float(row[idx_diag]))
    except:
        continue
    data.append({"image": img_name, "diagnosis": diagnosis, "gradable": gradable})

print(f"✅ Filtered gradable images: {len(data)} usable samples.")
# -----------------------------------------------------------------

# Create temp class folders
class_labels = [0, 1, 2, 3, 4]
temp_class_dir = "temp_messidor_class_folders"
if os.path.exists(temp_class_dir):
    print(f"ℹ️ Removing existing temp dir: {temp_class_dir}")
    shutil.rmtree(temp_class_dir)
os.makedirs(temp_class_dir, exist_ok=True)
for cls in class_labels:
    os.makedirs(os.path.join(temp_class_dir, f"class_{cls}"), exist_ok=True)

# ---------------- Copy images using robust path resolution ----------------
print("📸 Copying Messidor images by diagnosis class (using resolve_image_path)...")
copied_count = 0
missing_files = []
missing_samples_display = []

for rec in data:
    img_name = rec["image"].strip()
    diag = rec["diagnosis"]
    dest_folder = os.path.join(temp_class_dir, f"class_{diag}")
    os.makedirs(dest_folder, exist_ok=True)

    found = resolve_image_path(img_name, messidor_images)
    if found:
        try:
            shutil.copy(found, dest_folder)
            copied_count += 1
        except Exception as e:
            missing_files.append((img_name, [f"FOUND_BUT_COPY_FAILED: {found} ({e})"]))
            if len(missing_samples_display) < 10:
                missing_samples_display.append(f"{img_name} (copy failed)")
    else:
        tried = []
        tried.append(os.path.join(messidor_images, img_name))
        base, ext = os.path.splitext(img_name)
        if ext:
            for e in COMMON_EXTS:
                tried.append(os.path.join(messidor_images, base + e))
        for e in COMMON_EXTS:
            tried.append(os.path.join(messidor_images, img_name + e))
        csv_basename = os.path.splitext(img_name)[0].lower()
        try:
            for fname in os.listdir(messidor_images):
                if os.path.splitext(fname)[0].lower() == csv_basename:
                    tried.append(os.path.join(messidor_images, fname))
        except FileNotFoundError:
            pass

        tried_unique = []
        seen = set()
        for p in tried:
            if p not in seen:
                tried_unique.append(p)
                seen.add(p)

        missing_files.append((img_name, tried_unique))
        if len(missing_samples_display) < 10:
            missing_samples_display.append(tried_unique[0])

print(f"✅ Copied {copied_count} images to {temp_class_dir}. Missing files: {len(missing_files)}")
if missing_samples_display:
    print("⚠️ Sample missing files (first tried candidate shown):")
    for p in missing_samples_display[:10]:
        print("   -", p)

missing_debug_csv = os.path.join(output_folder, "missing_files_debug.csv")
with open(missing_debug_csv, "w", encoding="utf-8", newline="") as mf:
    writer = csv.writer(mf)
    writer.writerow(["csv_image_name", "attempted_candidates"])
    for img_name, candidates in missing_files:
        writer.writerow([img_name, ";".join(candidates)])
print(f"🔍 Detailed missing-file debug CSV written to: {missing_debug_csv}")
# ------------------------------------------------------------------------------

# --- Non-IID partitioning into clients ---
print("🔀 Creating non-IID client datasets with train/test splits...")
subfolders = sorted(os.listdir(temp_class_dir))
client_summary = {
    f"client_{i+1}": {f"class_{c}": {"train": 0, "test": 0} for c in class_labels}
    for i in range(num_clients)
}

for i in range(num_clients):
    client_folder = os.path.join(output_folder, f"client_{i+1}")
    os.makedirs(client_folder, exist_ok=True)

    for subfolder in subfolders:
        class_folder = os.path.join(temp_class_dir, subfolder)
        class_files = os.listdir(class_folder)
        random.shuffle(class_files)

        num_shards = min(num_shards_per_class, max(1, len(class_files)))
        local_class_temp = os.path.join(client_folder, subfolder, "local_temp")
        os.makedirs(local_class_temp, exist_ok=True)

        for shard in range(num_shards):
            if not class_files:
                break
            if random.random() < eta:
                k = min(num_samples_per_shard, len(class_files))
                shard_samples = random.sample(class_files, k)
            else:
                k = min(num_samples_per_shard, len(class_files))
                shard_samples = list(np.random.choice(class_files, size=k, replace=False))
            for sample in shard_samples:
                src = os.path.join(class_folder, sample)
                dst = os.path.join(local_class_temp, sample)
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy(src, dst)

        local_files = os.listdir(local_class_temp)
        random.shuffle(local_files)
        split_idx = int(len(local_files) * train_frac)
        train_files = local_files[:split_idx]
        test_files = local_files[split_idx:]

        train_dir = os.path.join(client_folder, subfolder, "train")
        test_dir = os.path.join(client_folder, subfolder, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for f in train_files:
            shutil.move(os.path.join(local_class_temp, f), os.path.join(train_dir, f))
        for f in test_files:
            shutil.move(os.path.join(local_class_temp, f), os.path.join(test_dir, f))

        client_summary[f"client_{i+1}"][subfolder]["train"] = len(os.listdir(train_dir))
        client_summary[f"client_{i+1}"][subfolder]["test"] = len(os.listdir(test_dir))
        try:
            os.rmdir(local_class_temp)
        except OSError:
            pass

print("🎉 Non-IID Messidor client datasets created with train/test splits.")
print("Clients are in:", output_folder)

# Optional server test set
if create_server_test:
    print(f"🧪 Creating server_test folder with {server_test_count} random Messidor images...")
    server_test_dir = os.path.join(output_folder, "server_test")
    os.makedirs(server_test_dir, exist_ok=True)

    all_files = []
    for sub in subfolders:
        cf = os.path.join(temp_class_dir, sub)
        for fname in os.listdir(cf):
            all_files.append(os.path.join(cf, fname))

    n_select = min(server_test_count, len(all_files))
    selected = random.sample(all_files, n_select)
    for src in selected:
        shutil.copy(src, server_test_dir)

    print(f"✅ Server test set created with {n_select} images at {server_test_dir}")

# Print summary
print("\n📋 Client dataset summary (train/test counts per class):")
for client, classes in client_summary.items():
    total_train = sum(info["train"] for info in classes.values())
    total_test = sum(info["test"] for info in classes.values())
    print(f"  - {client}: train={total_train}, test={total_test}")

print("\n✅ Done. All outputs are in:", output_folder)
