import os
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# -------------------------------------------
# CONFIG — edit these paths if needed
# -------------------------------------------
# A training client directory that reflects the same class-folder layout used for training.
# e.g. "isic_alpha_1.0/client_1" (any client folder is fine since folder names/order is global)
TRAIN_CLIENT_DIR = "isic_alpha_1.0/client_1"  

VAL_DIR = "ISIC2018_dataset/validation_data"
TEST_DIR = "ISIC2018_dataset/testing_data"

VAL_CSV = "validation_groundtruth.csv"
TEST_CSV = "testing_groundtruth.csv"

# This is the column order used in your CSV files (your current order)
CLASS_COLS = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

VAL_IMAGES_OUT = "val_images.npy"
VAL_LABELS_OUT = "val_labels.npy"              # will be overwritten with reordered labels if TRAIN_CLIENT_DIR available
TEST_IMAGES_OUT = "test_images.npy"
TEST_LABELS_OUT = "test_labels.npy"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
# -------------------------------------------


def detect_training_order(train_client_dir):
    """Use flow_from_directory to detect training class -> index mapping (model output index order)."""
    if not os.path.isdir(train_client_dir):
        print(f"[WARN] TRAIN_CLIENT_DIR not found: {train_client_dir}. Will NOT attempt to reorder labels.")
        return None

    datagen = ImageDataGenerator(rescale=1./255.)
    gen = datagen.flow_from_directory(
        train_client_dir,
        target_size=IMG_SIZE,
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    class_indices = gen.class_indices
    # Build ordered list where index -> class name
    ordered = [None] * len(class_indices)
    for cls, idx in class_indices.items():
        ordered[idx] = cls

    print("\n[INFO] Detected training class index -> class name mapping (model output index):")
    for i, c in enumerate(ordered):
        print(f"  Index {i}: {c}")

    return ordered


def reorder_labels_if_needed(labels, csv_class_cols, train_order):
    """
    Reorder labels (N, C) from csv_class_cols order to train_order.
    csv_class_cols: list of class names in same order as columns in labels (your CSV order).
    train_order: list of class names in training/model output order (index -> class).
    Returns reordered labels (or original if train_order is None).
    """
    if train_order is None:
        print("[INFO] No training order provided — keeping CSV label order.")
        return labels

    # sanity checks
    csv_set = set(csv_class_cols)
    train_set = set(train_order)
    if csv_set != train_set:
        print("[WARN] Class sets between CSV and training directory differ.")
        print(" CSV columns:", csv_class_cols)
        print(" Train order :", train_order)
        # Try to proceed with intersection: pick only shared classes
        common = list(csv_set.intersection(train_set))
        if len(common) == 0:
            print("[ERROR] No common classes — cannot reorder. Returning original labels.")
            return labels
        # Build CSV -> indices mapping then construct reorder mapping only for common classes
        reorder = []
        for cls in train_order:
            if cls in csv_class_cols:
                reorder.append(csv_class_cols.index(cls))
        print("[INFO] Partial reorder used for common classes:", reorder)
        return labels[:, reorder]

    # Build exact reorder mapping: for each train_index i, which CSV column index corresponds?
    reorder = [csv_class_cols.index(cls) for cls in train_order]
    print("\n[INFO] Reorder mapping (model_index -> csv_column_index):", reorder)
    print("Meaning: model output index 0 -> CSV column index", reorder[0], f"({train_order[0]})")
    return labels[:, reorder]


def process_dataset(data_dir, csv_name, out_images, out_labels, img_size=(224,224), train_order=None):
    """
    Loads images + one-hot labels and saves them to numpy files.
    If train_order is provided, label columns are reordered to match model output indices.
    """
    csv_path = os.path.join(data_dir, csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Class distribution from CSV (before missing-file filtering)
    print(f"\n=== CLASS COUNTS FROM {csv_name} ===")
    class_counts = df[CLASS_COLS].sum()
    print(class_counts)

    images = []
    labels = []

    missing = 0
    total = len(df)

    for _, row in df.iterrows():
        img_name = row['image'] + ".jpg"
        img_path = os.path.join(data_dir, img_name)

        if not os.path.exists(img_path):
            print(f"[WARN] Missing image: {img_path}")
            missing += 1
            continue

        # Load + preprocess image (kept as 0-255 ints; normalize later in training code)
        img = Image.open(img_path).convert("RGB").resize(img_size)
        images.append(np.array(img))

        # Load one-hot label in CSV order
        labels.append(row[CLASS_COLS].values.astype(float))

    images = np.array(images)
    labels = np.array(labels)

    # Reorder labels to match training/model output if train_order provided
    labels_reordered = reorder_labels_if_needed(labels, CLASS_COLS, train_order)

    # Save outputs
    np.save(out_images, images)
    np.save(out_labels, labels_reordered)

    print(f"\n[INFO] Processed {csv_name}:")
    print(f"  Total rows: {total}")
    print(f"  Missing images: {missing}")
    print(f"  Saved: {out_images} shape={images.shape}")
    print(f"  Saved: {out_labels} shape={labels_reordered.shape}")

    return images.shape, labels_reordered.shape, missing, total


if __name__ == "__main__":
    print("\n==================== Detect training order (if possible) ====================")
    train_order = detect_training_order(TRAIN_CLIENT_DIR)

    print("\n==================== VALIDATION SET ====================")
    process_dataset(
        VAL_DIR, VAL_CSV,
        VAL_IMAGES_OUT, VAL_LABELS_OUT,
        IMG_SIZE,
        train_order=train_order
    )

    print("\n==================== TESTING SET =======================")
    process_dataset(
        TEST_DIR, TEST_CSV,
        TEST_IMAGES_OUT, TEST_LABELS_OUT,
        IMG_SIZE,
        train_order=train_order
    )

    print("\n[ALL DONE] Validation and Testing numpy files created successfully!")
