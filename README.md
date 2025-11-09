---

# 🔥 Wildfire Spread Prediction using Deep U-Net

### 🧠 Project Overview

This project implements a **deep U-Net convolutional neural network** to predict **next-day wildfire spread** across the United States using multi-source environmental and geospatial data aggregated from 2012 – 2020.
Each training sample represents a **64 × 64 km** region (1 km resolution) with 12 input features describing weather, vegetation, and terrain conditions.
The model performs **pixel-wise segmentation** to forecast which grid cells are likely to ignite the following day.

---

## 🚀 Key Contributions

* 📦 Built a full **TensorFlow TFRecord data pipeline** for efficient loading, parsing, and normalization of 18 k+ wildfire samples.
* ⚙️ Designed a **deep U-Net** (encoder–decoder with skip connections) for spatial fire-mask prediction.
* 📉 Implemented **combined Binary Cross-Entropy + Dice loss** to address heavy class imbalance (few fire pixels).
* 📊 Achieved an average **IoU = 0.25–0.30** on held-out test data.
* 🎯 Visualized predicted vs true fire masks to evaluate spatial accuracy.

---

## 🗂️ Dataset Summary

| Feature Type       | Name                                  | Description / Units                                        | Effect on Fire Spread                       |
| ------------------ | ------------------------------------- | ---------------------------------------------------------- | ------------------------------------------- |
| **Meteorological** | `tmmn`, `tmmx`, `sph`, `pr`           | Temperature (K), Humidity, Precipitation                   | Control dryness & ignition likelihood       |
| **Environmental**  | `NDVI`, `pdsi`, `erc`                 | Vegetation index, Drought index, Energy release component  | Affect fuel availability & fire intensity   |
| **Geographical**   | `elevation`, `vs`, `th`, `population` | Terrain height, Wind speed & direction, Population density | Influence spread direction & human ignition |
| **Historical**     | `PrevFireMask`                        | Binary mask of previous fires                              | Indicates recent burn zones                 |
| **Target**         | `FireMask`                            | Binary next-day fire occurrence                            | Model output                                |

> Dataset Source: [Kaggle — Next Day Wildfire Spread (2012-2020)](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread)

---

## 🧩 Data Pipeline Architecture

```text
TFRecord files
   │
   ├── get_raw_dataset()     →  reads serialized TFRecords in parallel
   ├── parse_raw()           →  parses bytes → float tensors
   ├── normalize_feature()   →  standardizes each feature (μ, σ from train set)
   ├── _parse_function()     →  stacks 12 input layers → (64, 64, 12)
   └── get_dataset()         →  shuffle + batch + prefetch → train/val/test
```

### Normalization

[
x_{norm} = \frac{x - \mu}{\sigma + 10^{-6}}
]

* Computed per-feature using **training data only** to avoid leakage.
* `PrevFireMask` excluded (already binary).

---

## 🧱 Model Architecture — Deep U-Net (64×64)

| Stage      | Operation                                     | Output Shape         |
| ---------- | --------------------------------------------- | -------------------- |
| Encoder    | Conv → BN → ReLU ×2 + MaxPool                 | 64² → 32² → 16² → 8² |
| Bottleneck | Conv Block (8 × filters)                      | 8×8                  |
| Decoder    | Conv2D Transpose + Concat (skip) + Conv Block | 16² → 32² → 64²      |
| Output     | Conv (1×1) + Sigmoid                          | 64×64×1              |

* **Filters:** 32 → 64 → 128 → 256
* **Dropout:** encoder (0.10), bottleneck (0.20), decoder (0.10)
* **Regularization:** L2 = 1e-5

---

## ⚙️ Training Configuration

| Parameter  | Value                                              |
| ---------- | -------------------------------------------------- |
| Optimizer  | Adam (lr = 1e-3)                                   |
| Loss       | 0.5 × BCE + 0.5 × Dice                             |
| Metric     | Intersection over Union (IoU)                      |
| Batch Size | 32                                                 |
| Epochs     | 100                                                |
| Callbacks  | Early Stopping (patience = 15) & ReduceLROnPlateau |

---

## 🧮 Evaluation

| Metric                  | Test Result                                                                              |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| **IoU**                 | ≈ 0.30                                                                            |
| **Qualitative Results** | Model accurately highlights fire-prone areas; occasional false positives in dry regions. |

Example visualization:

|    Previous Day Fire    |       Ground Truth      |   Predicted Fire Mask   |
| :---------------------: | :---------------------: | :---------------------: |
| ![](docs/prev_fire.png) | ![](docs/true_mask.png) | ![](docs/pred_mask.png) |

---

## 📊 Inference Visualization

```python
show_inference(
    n_rows=3,
    features=sample_inputs,
    label=sample_labels,
    prediction_function=lambda x: model.predict(x)
)
```

---

## 🧰 Tech Stack

* **TensorFlow 2 / Keras** – model and data pipeline
* **NumPy / Matplotlib** – statistics and visualization
* **Google Colab** – GPU training environment
* **Python 3.8+**

---


### ⭐ Quick Summary for Interview

> “I built a full TFRecord → TensorFlow pipeline for wildfire segmentation.
> It reads 64×64 spatial grids, normalizes 12 environmental features,
> and trains a deep U-Net using BCE + Dice loss to predict next-day fire spread with IoU ≈ 0.3.”


