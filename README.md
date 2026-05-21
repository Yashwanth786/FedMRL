# Federated Multi-Agent Reinforcement Learning for Medical Imaging

### *Reproduction of FedMRL + Proposed GNN-Guided QMIX Meta-Aggregator*

This project reproduces **FedMRL: Data Heterogeneity Aware Federated Multi-Agent Deep Reinforcement Learning for Medical Imaging** and proposes a novel extension using **Graph Neural Networks (GNNs)** to guide aggregation weights based on inter-client similarity for highly non-IID federated medical datasets.

---

# 1. Project Overview

Federated learning enables collaborative model training without centralizing hospital data—but suffers when data is **non-IID, class-imbalanced, and distribution-shifted** across institutions. FedMRL addresses this through:

| Component | Purpose |
|----------|---------|
| **Fed Learning (FedAvg Base)** | Distributed training without sharing data |
| **MARL via QMIX** | Learns client-specific proximal term μ dynamically |
| **Fairness-Aware Loss** | Reduces client-level accuracy disparity |
| **SOM-Based Aggregation** | Weights clients by similarity to global model |

##  Proposed Improvement

I introduce a **GNN-Guided Meta Aggregator**, where:
* Clients = graph nodes
* Similarity of gradients = edges
* **GNN learns aggregation weights** ----> fed into QMIX

This improves fairness and performance on highly skewed datasets (especially Messidor).

---

# 2. Architecture Diagram

```markdown
(Architecture.png)
```
---
# 3. Abstract

Federated learning offers privacy-preserving collaboration across hospitals, yet real-world medical datasets are highly non-IID and class-imbalanced, reducing effectiveness of standard averaging-based methods. FedMRL integrates multi-agent reinforcement learning with fairness-aware loss optimization and SOM-based aggregation to mitigate heterogeneity across clients.

This project reproduces FedMRL using MobileNetV2 to address hardware limitations and evaluates performance on ISIC 2018 and Messidor datasets. We further introduce a GNN-guided QMIX meta-aggregator that models inter-client similarity as graphs and learns dynamic aggregation weights. Results show marginal improvements on ISIC but significant performance and balanced accuracy gains on Messidor, demonstrating the importance of graph-aware aggregation under extreme distributional shift.

---

# 4. Dataset Setup

This project uses two medical imaging datasets stored locally due to privacy and size constraints.

| Dataset | Task | Classes | Size | Source |
|---------|------|----------|------|--------|
| **ISIC 2018** | Skin lesion classification | 7 classes | ~3.1GB | ISIC Challenge Archive |
| **Messidor 2** | Diabetic Retinopathy Grading | 5 classes | ~1GB | Kaggle |



## **4.1 ISIC2018 Dataset Setup**

Dataset link: **https://challenge.isic-archive.com/data/#2018**

You need to download the following **six files manually**:

| Component | Description | Size |
|-----------|------------|------|
| Training Input Images | 10,015 dermoscopy images | ~2.6GB |
| Training Ground Truth CSV | Labels for training set | ~36 KB |
| Validation Input Images | 195 images | ~51MB |
| Validation Ground Truth CSV | Labels for validation set | ~7 KB |
| Test Input Images | 1,512 test images | ~401MB |
| Test Ground Truth CSV | Labels for testing set | ~11 KB |

###  Required Folder Structure

* Parent Directory : **ISIC2018_dataset**

Create this directory structure:
```
fedmrl-main/
│── isic/
│   └── ISIC2018_dataset/
│       ├── training_data/
│       │   ├── <images>
│       │   └── training_groundtruth.csv
│       ├── validation_data/
│       │   ├── <images>
│       │   └── validation_groundtruth.csv
│       └── testing_data/
│           ├── <images>
│           └── testing_groundtruth.csv

```


## **4.2 Messidor Dataset Setup**

Dataset link: **https://www.kaggle.com/datasets/mariaherrerot/messidor2preprocess/data**

Messidor contains one download package containing **1744** images + CSV metadata.


###  Required Folder Structure

* Parent Directory : **Messidor_dataset**

Create this directory structure:
```
fedmrl-main/
│── messidor/
│   └── Messidor_dataset/
│       ├── messidor-2/          ← images folder
│       └── messidor_data.csv    ← metadata

```

## Note:
* **Do NOT rename images**  .
* Rename CSVs and folders exactly as shown.
* Place the datasets in their respective folders:
    * **isic / ISIC2018_dataset**
    * **messidor / Messidor_dataset**

---

# 5. Create Virtual Environment & Install Dependencies

Run these commands in parent directory.

### Windows (CMD/PowerShell)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### macOS / Linux
```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

### Verify Installation
```
python --version
pip list
```

---

# 6. Implementation — ISIC Dataset

This section explains how to run the full pipeline for **ISIC 2018**, including data preparation, model training, GNN-based training, and evaluation.


## 📍 Step 1 — Change Directory

Navigate into the **ISIC project folder**:

```bash
cd isic
```
Directory should look like:
```
fedmrl-main/
│── isic/
│   ├── ISIC2018_dataset/
│   ├── non_iid_data_preparation_isic.py
│   ├── prepare_test_arrays_isic.py
│   ├── model_isic.py
│   ├── fedmrl_isic.py
│   ├── fedmrl_gnn.py
│   ├── Evaluation_isic.py
```
---

## 📍 Step 2 — Create Non-IID Client Partitions

This script partitions the dataset into 4 clients, with 7 classes per client stored under a new folder:

```
python non_iid_data_preparation_isic.py
```

### Output:

```
isic_alpha_1.0/
│── client_1/
│   ├── class_0/
│   ├── class_1/
│   ...
│── client_4/
```
---

## 📍 Step 3 — Prepare Test Arrays (One-hot Labels + Preprocessed Images)
```
python prepare_test_arrays_isic.py
```
### Output files:
```
val_images.npy
val_labels.npy
test_images.npy
test_labels.npy
```

Stored in current directory.

---

## 📍 Step 4 — Load Pretrained Model (MobileNetV2)
```
python model_isic.py
```

This script defines the feature extractor architecture and classification head.

---

## 📍 Step 5 — Train FedMRL (Baseline — No GNN)

Run the main training script:
```
python fedmrl_isic.py
```

When prompted:
```
Which gpu number would you like to allocate (0, 1, etc., or -1 for CPU):
```

Example:
```
0            → GPU 0
1            → GPU 1
-1           → CPU mode
```
### Training details:

| Setting              | Value           |
| -------------------- | --------------- |
| Epochs               | 60              |
| Time per epoch (CPU) | ~160 sec        |
| Output folder        | `saved_models/` |

---

## 📍Step 6 — Train FedMRL + GNN (Proposed Method)
```
python fedmrl_gnn.py
```

Same GPU input prompt as Step 5.

### Training details:

| Setting              | Value               |
| -------------------- | ------------------- |
| Epochs               | 60                  |
| Time per epoch (CPU) | ~164 sec            |
| Output folder        | `saved_models_gnn/` |

---

## 📍 Step 7 — Evaluation
```
python Evaluation_isic.py
```

#### Menu prompt:
```
Select model type:
0 → saved_models
1 → saved_models_gnn
```

### Output

| Model Type    | Folder                   |
| ------------- | ------------------------ |
| Normal FedMRL | `evaluation_report/`     |
| FedMRL + GNN  | `evaluation_report_gnn/` |

---

## Results (ISIC)

| Metric                    | FedMRL (Baseline) | FedMRL + GNN (Proposed) |
| ------------------------- | ----------------- | ----------------------- |
| **Accuracy (%)**          | **72.54**         | 66.87                   |
| **Precision (%)**         | **76.65**         | 68.55                   |
| **Recall (%)**            | **72.54**         | 66.86                   |
| **F1-score (%)**          | **74.14**         | 66.10                   |
| **Balanced Accuracy (%)** | 41.29             | **42.12**               |
| **Kappa**                 | **0.53**          | 0.44                    |
| **MCC**                   | **0.53**          | 0.44                    |
| **AUC (%)**               | **92.27**         | 86.63                   |


---

# 7. Implementation — Messidor Dataset

This section describes how to run the full workflow for Messidor, including data preparation, training (baseline + GNN), and evaluation.

## 📍 Step 1 — Change Directory
Navigate into the Messidor project folder:
```
cd messidor
```

Folder structure should look like:
```
fedmrl-main/
│── messidor/
│   ├── Messidor_dataset/
│   ├── non_iid_data_preparation_messidor.py
│   ├── prepare_test_arrays_messidor.py
│   ├── model_messidor.py
│   ├── fedmrl_messidor.py
│   ├── fedmrl_gnn.py
│   ├── Evaluation_messidor.py
```
---

## 📍 Step 2 — Create Non-IID Client Partitions

This script creates 4 clients, 5 classes, and each class is internally split 80:20 (train:test).
```
python non_iid_data_preparation_messidor.py
```

### Output folder structure:
```
messidor_alpha_1.0/
│── client_1/
│   ├── class_0/
│   │   ├── train/
│   │   └── test/
│   └── class_4/
│── client_4/
```
---


## 📍 Step 3 — Prepare Test Arrays

Creates one-hot encoded labels + NumPy image arrays.
```
python prepare_test_arrays_messidor.py
```

### Output files:
```
test.npy
one_hot_labels.npy
```

Stored in current directory.

---


## 📍 Step 4 — Load Pretrained MobileNetV2 Model
```
python model_messidor.py
```

Defines feature extractor + classification head.

---

## 📍 Step 5 — Train FedMRL (Baseline Model)
```
python fedmrl_messidor.py
```

Same GPU input prompt as ISIC implementaion.

### Training details:

| Property             | Value           |
| -------------------- | --------------- |
| Epochs               | 60              |
| Time per epoch (CPU) | ~164 sec        |
| Output folder        | `saved_models/` |

---

## 📍 Step 6 — Train FedMRL + GNN (Proposed Method)
```
python fedmrl_gnn.py
```
Same GPU input prompt as above step 5.

### Training details:

| Property             | Value               |
| -------------------- | ------------------- |
| Epochs               | 60                  |
| Time per epoch (CPU) | ~170 sec            |
| Output folder        | `saved_models_gnn/` |

---

## 📍 Step 7 — Evaluation
```
python Evaluation_messidor.py
```

#### Program prompt:
```
Select model type:
0 → saved_models
1 → saved_models_gnn
```

###  Output folders:

| Model Type      | Folder                   |
| --------------- | ------------------------ |
| Baseline FedMRL | `evaluation_report/`     |
| FedMRL + GNN    | `evaluation_report_gnn/` |

---

## Results (Messidor)

| Metric                    | FedMRL (Baseline) | FedMRL + GNN (Proposed) |
| ------------------------- | ----------------- | ----------------------- |
| **Accuracy (%)**          | 59.86             | **72.36**               |
| **Precision (%)**         | 52.45             | **71.90**               |
| **Recall (%)**            | 59.86             | **72.36**               |
| **F1-score (%)**          | 52.38             | **69.03**               |
| **Balanced Accuracy (%)** | 25.46             | **47.89**               |
| **Kappa**                 | 0.18              | **0.47**                |
| **MCC**                   | 0.20              | **0.50**                |
| **AUC (%)**               | 72.35             | **88.98**               |

---
