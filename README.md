# Knee MRI Abnormality Detection

Deep learning model for automated detection of multiple knee abnormalities from MRI studies.

The model processes multiple MRI series and multiple slices per series using a ResNet50 backbone combined with a two-stage Attention-based Multiple Instance Learning (MIL) architecture.

---

## Overview

A knee MRI study contains multiple imaging series, and each series consists of multiple 2D MRI slices.

Instead of analyzing a single MRI image independently, this model processes the study hierarchically:

MRI Study
↓
Relevant MRI Series
↓
16 Slices per Series
↓
ResNet50 Feature Extraction
↓
Slice-Level Attention
↓
Series Representations
↓
Series-Level Attention
↓
Study Representation
↓
12 Abnormality Predictions

The model performs multi-label classification, meaning that multiple abnormalities can be predicted for the same MRI study.

---

## Abnormalities

The model predicts the following 12 targets:

1. ACL
2. MCL
3. Medial Meniscus
4. Lateral Meniscus
5. Medial Osteoarthritis
6. Lateral Osteoarthritis
7. Patellofemoral Osteoarthritis
8. Effusion
9. Synovitis
10. Baker's Cyst
11. Contusion
12. Fracture

Each output is an independent probability between 0 and 1.

Example:

```text
ACL              0.86
MCL              0.12
Medial Meniscus  0.79
Lateral Meniscus 0.18
Medial OA        0.91
...
