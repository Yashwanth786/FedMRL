import os
import random
import shutil
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict

# -------------------------
# PARAMETERS (same as original code)
# -------------------------
images_folder = os.path.join("ISIC2018_dataset", "training_data")
csv_path = os.path.join("ISIC2018_dataset", "training_data", "training_groundtruth.csv")

output_folder = "isic_alpha_1.0"

num_clients = 4
num_shards_per_class = 200
eta = 1.0                      # Probability of sampling from same class
num_samples_per_shard = 5      # How many images per shard
random.seed(42)

# ---------------------------------------------------
# 1. Read CSV and prepare class → image list mapping
# ---------------------------------------------------
df = pd.read_csv(csv_path)
class_names = [c for c in df.columns if c != "image"]
print("Classes found:", class_names)

# One-hot to class mapping
class_to_files = OrderedDict((c, []) for c in class_names)

def find_image(image_id):
    """Return real path to image ISIC_xxx.ext inside training_input folder."""
    for ext in [".jpg", ".jpeg", ".png"]:
        p = os.path.join(images_folder, image_id + ext)
        if os.path.exists(p):
            return p
    # fallback to partial match
    for f in os.listdir(images_folder):
        if f.startswith(image_id):
            return os.path.join(images_folder, f)
    return None

missing = []

for _, row in df.iterrows():
    image_id = str(row["image"])
    class_vals = row[class_names].values.astype(float)
    if np.all(class_vals == 0):
        continue
    class_name = class_names[int(np.argmax(class_vals))]
    img_path = find_image(image_id)

    if img_path:
        class_to_files[class_name].append(img_path)
    else:
        missing.append(image_id)

if missing:
    print(f"Warning: {len(missing)} images missing from folder.")

# ---------------------------------------------------
# 2. Print global counts
# ---------------------------------------------------
print("\nTotal images per class:")
for cname, flist in class_to_files.items():
    print(f"  {cname}: {len(flist)}")
print()

# ---------------------------------------------------
# 3. Create Non-IID Shard-Based Clients (your original logic)
# ---------------------------------------------------
os.makedirs(output_folder, exist_ok=True)

# Store counts in dict
client_class_counts = {
    f"client_{i+1}": {c: 0 for c in class_names}
    for i in range(num_clients)
}

for client_idx in range(num_clients):
    client_name = f"client_{client_idx+1}"
    client_folder = os.path.join(output_folder, client_name)
    os.makedirs(client_folder, exist_ok=True)

    print(f"\nCreating data for {client_name}...")

    for class_name in class_names:
        class_folder = os.path.join(client_folder, class_name)
        os.makedirs(class_folder, exist_ok=True)

        class_files = class_to_files[class_name].copy()
        random.shuffle(class_files)
        total_class_files = len(class_files)

        # Limit shards if dataset is small
        num_shards = min(num_shards_per_class, total_class_files)

        for shard in range(num_shards):
            
            # Decide whether sampling from same class or other classes
            sample_same_class = (random.random() < eta)

            if sample_same_class:
                # Same class sampling
                chosen = random.sample(
                    class_files,
                    min(num_samples_per_shard, len(class_files))
                )
            else:
                # Sample from other classes
                other_classes = [c for c in class_names if c != class_name]
                other_class = random.choice(other_classes)
                other_files = class_to_files[other_class]
                chosen = random.sample(
                    other_files,
                    min(num_samples_per_shard, len(other_files))
                )

            # Copy files
            for src in chosen:
                dst = os.path.join(class_folder, os.path.basename(src))
                shutil.copy(src, dst)
                client_class_counts[client_name][class_name] += 1

# ---------------------------------------------------
# 4. Print summary
# ---------------------------------------------------
print("\n==========================")
print("SUMMARY: Images per class per client")
print("==========================\n")

summary_df = []

for client in client_class_counts:
    print(client, "→", client_class_counts[client])
    row = {"client": client}
    row.update(client_class_counts[client])
    summary_df.append(row)

summary_df = pd.DataFrame(summary_df)
summary_df.to_csv(os.path.join(output_folder, "client_class_summary.csv"), index=False)

print(f"\nSaved summary to {output_folder}/client_class_summary.csv")
