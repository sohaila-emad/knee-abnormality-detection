# Multi-Series Attention MIL for Knee MRI Abnormality Detection

An end-to-end deep learning framework designed to detect **12 distinct knee joint abnormalities** simultaneously from multi-series DICOM MRI studies. Utilizing a 2D ResNet-50 CNN feature extractor coupled with a two-stage hierarchical Attention-based Multiple Instance Learning (MIL) architecture, the model aggregates slice-level and series-level information to make study-level diagnostic predictions.

---

## Architecture Overview

Medical imaging studies contain a high degree of spatial and structural variance. A standard knee MRI consists of multiple series (imaging angles/sequences, such as Sagittal T2, Coronal T1, Axial PD, etc.), where each series contains anywhere from dozens to hundreds of 2D slices.

Instead of processing slices independently or relying on dense 3D convolutions that require massive GPU memory, this model processes studies hierarchically:

```text
               Knee MRI Study
                     │
         ┌───────────┴───────────┐
      Series 1                Series N  (Multiple sequences/angles)
         │                       │
 ┌───────┴───────┐       ┌───────┴───────┐
Slice 1 ... Slice 16   Slice 1 ... Slice 16  (16 sampled DICOM slices per series)
 └───────┬───────┘       └───────┬───────┘
         │                       │
   [ResNet-50]             [ResNet-50]       (Feature Extraction)
         │                       │
 [Slice-Level MIL]       [Slice-Level MIL]   (Slice Attention Pooling)
         │                       │
   Series Rep 1            Series Rep N      (512-dim Feature Embeddings)
         └───────────┬───────────┘
                     │
             [Series-Level MIL]              (Series Attention Pooling)
                     │
             Study Embedding
                     │
          [Sigmoid Classification]
                     │
          12 Target Predictions              (Independent Probabilities [0.0 - 1.0])

```

1. **Slice Sampling & Preprocessing:** Standardizes inputs by sorting DICOM slices anatomically and uniformly extracting $K = 16$ representative 2D slices per series.
2. **Feature Extraction:** A shared 2D ResNet-50 backbone extracts spatial feature embeddings from individual 2D MRI slices.
3. **Stage 1 (Slice-Level Attention MIL):** An attention mechanism weights the most clinically relevant slices within a series to produce a single 512-dimensional vector representation per series.
4. **Stage 2 (Series-Level Attention MIL):** A second-stage attention network pools feature embeddings across all relevant series within a study, producing a unified Study Representation.
5. **Multi-Label Classification:** A final linear head predicts independent sigmoid probabilities across all 12 target pathologies.

---

## Targets & Predictions

The model performs **multi-label classification**, predicting independent probabilities ($p \in [0, 1]$) for the following 12 conditions:

| # | Abnormality Target | Description |
| --- | --- | --- |
| **1** | **ACL** | Anterior Cruciate Ligament tear/abnormality |
| **2** | **MCL** | Medial Collateral Ligament tear/abnormality |
| **3** | **Medial Meniscus** | Medial meniscus tear or structural injury |
| **4** | **Lateral Meniscus** | Lateral meniscus tear or structural injury |
| **5** | **Medial Osteoarthritis** | Cartilage loss / degeneration in medial compartment |
| **6** | **Lateral Osteoarthritis** | Cartilage loss / degeneration in lateral compartment |
| **7** | **Patellofemoral Osteoarthritis** | Degeneration in the patellofemoral joint |
| **8** | **Effusion** | Joint effusion / fluid accumulation |
| **9** | **Synovitis** | Synovial membrane inflammation |
| **10** | **Baker's Cyst** | Popliteal fluid collection |
| **11** | **Contusion** | Bone marrow edema / bone bruise |
| **12** | **Fracture** | Bone cortical disruption / fracture line |

### Example Output

```text
ACL                       0.86
MCL                       0.12
Medial Meniscus           0.79
Lateral Meniscus          0.18
Medial Osteoarthritis     0.91
Lateral Osteoarthritis    0.05
Patellofemoral OA         0.22
Effusion                  0.88
Synovitis                 0.41
Baker's Cyst              0.03
Contusion                 0.65
Fracture                  0.01

```

---

## Data Ingestion & Sample Code

The dataset is structured inside Kaggle's environment with DICOM files stored per study and series:

```text
train_series/
└── <STUDY_UID>/
    ├── <SERIES_UID_1>/
    │   ├── 0001.dcm
    │   ├── 0002.dcm
    │   └── ...
    └── <SERIES_UID_2>/
        └── ...

```

### Loading and Visualizing DICOM Series

Use the snippet below to read DICOM series files, sort them along anatomical positions, and plot sampled slices across a sequence:

```python
import pathlib
import pydicom
import numpy as np
import matplotlib.pyplot as plt

# 1. Point to your train_series directory
TRAIN_SERIES_DIR = pathlib.Path("/kaggle/input/competitions/rsna-knee-abnormality-detection/train_series")

# Grab the first available study folder
study_folders = [f for f in TRAIN_SERIES_DIR.iterdir() if f.is_dir()]
sample_study = study_folders[0]
print(f"Viewing Study ID: {sample_study.name}")

# List all series folders inside this study
series_folders = [f for f in sample_study.iterdir() if f.is_dir()]
print(f"Found {len(series_folders)} series folders in this study.")

# 2. Helper function to load and sort DICOM slices into a 3D NumPy array
def load_series(series_path):
    dcm_files = list(series_path.glob("*.dcm"))

    # Read each DICOM file
    slices = [pydicom.dcmread(f) for f in dcm_files]

    # Sort slices in anatomical order using InstanceNumber or spatial position
    slices.sort(
        key=lambda s: int(s.InstanceNumber) 
        if 'InstanceNumber' in s 
        else float(s.ImagePositionPatient[2])
    )

    # Extract raw 2D pixel arrays
    images = np.array([s.pixel_array for s in slices])
    return images, slices[0]

# 3. Plot evenly-spaced slices from the first series
series_images, sample_header = load_series(series_folders[0])
print(f"Loaded Series Shape: {series_images.shape} -> ({series_images.shape[0]} Slices, {series_images.shape[1]}x{series_images.shape[2]} pixels)")

# Display 8 evenly spaced slices across the 3D volume
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
slice_indices = np.linspace(0, len(series_images) - 1, 8, dtype=int)

for idx, ax in zip(slice_indices, axes.flat):
    ax.imshow(series_images[idx], cmap="gray")
    ax.set_title(f"Slice {idx + 1}/{len(series_images)}")
    ax.axis("off")

plt.suptitle(f"Series: {series_folders[0].name}", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

```

---

## Requirements & Environment

* **Python:** 3.10+ (Kaggle Notebook Python 3.12 compatible)
* **Key Libraries:**
* `pydicom`
* `numpy`
* `matplotlib`
* `torch` & `torchvision` (ResNet-50 pretrained weights)
* `pandas` / `scikit-learn`



---

## Link & Dataset Access

* **Dataset & Kaggle Notebook:** [RSNA Knee Abnormality Detection Dataset / Kaggle Notebook Link](https://kaggle.com/datasets/f5b19fa832f184bf715c62947527871a4afd82e7c81372231bcb23fe0c05f6de)
