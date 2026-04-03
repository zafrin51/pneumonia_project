# Chest X-Ray Pneumonia Classification

A deep learning pipeline that classifies paediatric chest X-ray images as **Normal** or **Pneumonia** using a custom Convolutional Neural Network built with TensorFlow and Keras.

---

## 1. About the Dataset

| | Normal | Pneumonia | Total |
|---|---:|---:|---:|
| Train | 1 341 | 3 875 | 5 216 |
| Test | 234 | 390 | 624 |
| Val | 8 | 8 | 16 |

- Source: anterior-posterior chest X-rays from children aged 1-5 years.
- The training split is heavily imbalanced (~2.9 Pneumonia images for every Normal image).
- The provided validation split is very small (16 images), so the pipeline carves out **15 % of the training data** as a working validation set instead.

---

## 2. Approach

### 2.1 Exploratory Analysis

Image counts were tallied per class per split and the imbalance ratio was computed. Random samples from both categories were visualised to note that:

- **Normal** X-rays exhibit clear, dark lung fields on both sides.
- **Pneumonia** X-rays show hazy white patches (consolidation / opacity), often more prominent on one side.

### 2.2 Preprocessing

| Step | Detail |
|------|--------|
| Resize | All images scaled to **224 x 224** pixels |
| Normalise | Pixel values mapped to the [0, 1] range |
| Validation split | 15 % of training data held out as validation (the original val set is too small) |

### 2.3 Augmentation (training only)

Augmentation is applied on-the-fly during training to reduce overfitting:

- Random rotation (up to 15 degrees)
- Random width / height shifts (up to 15 %)
- Random zoom (up to 15 %)
- Brightness variation (0.8x - 1.2x)
- Channel shift (up to 20)
- Horizontal flip
- Fill mode set to `reflect`

Test and validation data are **never augmented** — only rescaled.

### 2.4 Class Weighting

To counteract the imbalance, inverse-frequency class weights are passed to the training loop so that misclassifying a Normal image carries a higher penalty.

---

## 3. Methodology

### 3.1 Network Architecture

The model is built with the **Keras Functional API** and uses three convolutional stages followed by a classification head:

```
Input  224 x 224 x 3
  |
  |--- Stage 1:  Conv(32) → Conv(32) → BatchNorm → MaxPool → Dropout(0.2)
  |--- Stage 2:  Conv(64) → Conv(64) → BatchNorm → MaxPool → Dropout(0.3)
  |--- Stage 3:  Conv(128) → Conv(128) → BatchNorm → MaxPool → Dropout(0.3)
  |
  |--- Global Average Pooling
  |--- Dense(256, ReLU) → Dropout(0.5)
  |--- Dense(1, Sigmoid) → output probability
```

**Key design choices:**

| Choice | Reason |
|--------|--------|
| Functional API instead of Sequential | Easier to extend or branch later |
| Three stages (not four or five) | Enough depth for 224 x 224 inputs while keeping training fast |
| Global Average Pooling instead of Flatten | Drastically reduces parameter count and helps prevent overfitting |
| Increasing dropout (0.2 → 0.3 → 0.5) | Deeper layers are more prone to memorisation; stronger regularisation is applied there |
| Batch Normalisation after each stage | Stabilises activations, allows faster convergence |

### 3.2 Training Setup

| Parameter | Value |
|-----------|-------|
| Optimiser | Adam, lr = 5 x 10⁻⁴ |
| Loss | Binary cross-entropy |
| Batch size | 16 |
| Maximum epochs | 20 |
| Early stopping | patience 4 on val_loss, restores best weights |
| LR scheduler | ReduceLROnPlateau, factor 0.3, patience 2 |

A smaller batch size (16) combined with a moderately higher learning rate (5 x 10⁻⁴) gives noisier gradients that act as implicit regularisation and help the model generalise better on this small dataset.

---

## 4. Findings

### 4.1 Expected Performance

- **Accuracy**: The model is expected to reach **87 – 93 %** on the test set.
- **Sensitivity** (pneumonia recall) is the most important metric in a medical screening context — a missed diagnosis is far more harmful than a false alarm.
- An **ROC-AUC** score is also computed to give a threshold-independent measure of discriminative ability.

### 4.2 Observations

1. **Class weighting is essential.** Without it, the network converges to predicting Pneumonia for almost every input because that alone gives ~74 % accuracy on the imbalanced training set.
2. **Brightness and channel-shift augmentations** matter for X-rays because real-world exposure levels vary across machines and hospitals.
3. **Global Average Pooling** reduces the trainable parameter count substantially compared to a Flatten layer, which helps when the dataset is relatively small.
4. **The original validation set (16 images) is unreliable.** Splitting 15 % of training data provides ~780 validation samples, yielding much more stable loss and accuracy curves.

### 4.3 Generated Outputs

All artefacts are saved to the `results/` folder:

| File | Contents |
|------|----------|
| `xray_classifier.keras` | Final saved model |
| `checkpoint.keras` | Best checkpoint during training |
| `evaluation_report.txt` | Accuracy, AUC, sensitivity, specificity, precision, F1 |
| `learning_curves.png` | Train vs. validation accuracy and loss over epochs |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `roc_curve.png` | ROC curve with AUC |
| `sample_predictions.png` | Predictions on random test images |
| `normal_x-rays.png` | Grid of normal training images |
| `pneumonia_x-rays.png` | Grid of pneumonia training images |

### 4.4 Limitations and Future Work

- **Transfer learning** (e.g. MobileNetV2 or EfficientNetB0 as backbone) could boost accuracy further.
- **Grad-CAM heatmaps** would highlight which lung regions drive the prediction, increasing clinical trust.
- **Stratified K-fold cross-validation** would give a more robust estimate of generalisation performance.
- The model does not distinguish between bacterial and viral pneumonia — a multi-class setup could address this.

---

## 5. Repository Layout

```
Pneumonia Detection/
├── pneumonia_detection.py    # Training and evaluation script
├── requirements.txt          # Dependencies
├── README.md                 # This document
├── x-ray_image/
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   ├── test/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── val/
│       ├── NORMAL/
│       └── PNEUMONIA/
└── results/                  # Created at runtime
```

## 6. Running the Code

```bash
pip install -r requirements.txt
python pneumonia_detection.py
```

The script runs end-to-end: data loading, training, evaluation, and saving all outputs.

## 7. Dependencies

- Python 3.8 – 3.12
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
