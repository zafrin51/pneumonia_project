"""
Chest X-Ray Classification: Detecting Pneumonia with Deep Learning
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks

warnings.filterwarnings('ignore')
np.random.seed(101)
tf.random.set_seed(101)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(SCRIPT_DIR, 'x-ray_image')

PATHS = {
    'train': os.path.join(DATASET_ROOT, 'train'),
    'test': os.path.join(DATASET_ROOT, 'test'),
    'val': os.path.join(DATASET_ROOT, 'val'),
}

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Hyperparameters
TARGET_SIZE = (224, 224)
BATCH = 16
MAX_EPOCHS = 20
LR = 5e-4


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def count_images(directory):
    """Return the number of files in a directory."""
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])


def dataset_summary():
    """Display a summary table of the dataset split."""
    print("\n" + "#" * 55)
    print("#   CHEST X-RAY DATASET SUMMARY")
    print("#" * 55)

    grand_total = 0
    for split_name, split_path in PATHS.items():
        n_normal = count_images(os.path.join(split_path, 'NORMAL'))
        n_pneumonia = count_images(os.path.join(split_path, 'PNEUMONIA'))
        total = n_normal + n_pneumonia
        grand_total += total
        ratio = n_pneumonia / n_normal if n_normal else 0

        print(f"\n  [{split_name.upper()}]")
        print(f"    Normal ......... {n_normal:>5}")
        print(f"    Pneumonia ...... {n_pneumonia:>5}")
        print(f"    Total .......... {total:>5}")
        print(f"    Imbalance ratio  {ratio:.2f}:1")

    print(f"\n  Grand Total: {grand_total} images")
    print("#" * 55 + "\n")


def show_grid(folder, title, rows=2, cols=4):
    """Display a grid of images from a folder."""
    files = os.listdir(folder)[:rows * cols]
    fig, axes = plt.subplots(rows, cols, figsize=(14, 6))
    fig.suptitle(title, fontsize=15, fontweight='bold')

    for i, fname in enumerate(files):
        r, c = divmod(i, cols)
        img = keras.preprocessing.image.load_img(
            os.path.join(folder, fname), target_size=TARGET_SIZE
        )
        axes[r, c].imshow(img, cmap='bone')
        axes[r, c].axis('off')

    plt.tight_layout()
    safe_title = title.replace(' ', '_').replace(':', '').lower()
    plt.savefig(os.path.join(RESULTS_DIR, f'{safe_title}.png'), dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------
def build_data_pipeline():
    """
    Prepare training, validation and test generators.

    Training data is augmented with brightness adjustment, channel shift
    and vertical flip in addition to the standard geometric transforms.
    A portion of training data (15 %) is reserved for validation because
    the provided validation split is too small (only 16 images).
    """
    train_aug = keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.15,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        channel_shift_range=20,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='reflect'
    )

    eval_scale = keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

    print(">> Preparing data generators ...")

    train_iter = train_aug.flow_from_directory(
        PATHS['train'],
        target_size=TARGET_SIZE,
        batch_size=BATCH,
        class_mode='binary',
        subset='training',
        shuffle=True,
        seed=101
    )

    val_iter = train_aug.flow_from_directory(
        PATHS['train'],
        target_size=TARGET_SIZE,
        batch_size=BATCH,
        class_mode='binary',
        subset='validation',
        shuffle=False,
        seed=101
    )

    test_iter = eval_scale.flow_from_directory(
        PATHS['test'],
        target_size=TARGET_SIZE,
        batch_size=BATCH,
        class_mode='binary',
        shuffle=False
    )

    print(f"   Training samples:   {train_iter.samples}")
    print(f"   Validation samples: {val_iter.samples}")
    print(f"   Test samples:       {test_iter.samples}")
    print(f"   Label map:          {train_iter.class_indices}\n")

    return train_iter, val_iter, test_iter


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------
def create_classifier():
    """
    Build a CNN using the Keras Functional API.

    The network has three convolutional stages. Each stage doubles the filter
    count and uses a pair of 3x3 convolutions followed by a 2x2 max-pool.
    Global Average Pooling replaces Flatten to reduce parameters and
    improve spatial invariance.
    """
    inp = keras.Input(shape=(*TARGET_SIZE, 3), name='xray_input')

    # Stage 1
    x = layers.Conv2D(32, 3, padding='same', activation='relu', name='conv1a')(inp)
    x = layers.Conv2D(32, 3, padding='same', activation='relu', name='conv1b')(x)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.MaxPooling2D(name='pool1')(x)
    x = layers.Dropout(0.2, name='drop1')(x)

    # Stage 2
    x = layers.Conv2D(64, 3, padding='same', activation='relu', name='conv2a')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu', name='conv2b')(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.MaxPooling2D(name='pool2')(x)
    x = layers.Dropout(0.3, name='drop2')(x)

    # Stage 3
    x = layers.Conv2D(128, 3, padding='same', activation='relu', name='conv3a')(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu', name='conv3b')(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.MaxPooling2D(name='pool3')(x)
    x = layers.Dropout(0.3, name='drop3')(x)

    # Head
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(256, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.5, name='drop_fc')(x)
    out = layers.Dense(1, activation='sigmoid', name='prediction')(x)

    model = keras.Model(inputs=inp, outputs=out, name='PneumoniaNet')
    return model


def compile_and_summarise(model):
    """Compile the model and print its architecture."""
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    print("\n>> Model architecture")
    model.summary()
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def compute_weights(generator):
    """Calculate class weights to offset training-set imbalance."""
    counts = np.bincount(generator.classes)
    total = generator.samples
    w = {i: total / (len(counts) * c) for i, c in enumerate(counts)}
    print(f">> Class weights: NORMAL={w[0]:.3f}  PNEUMONIA={w[1]:.3f}")
    return w


def fit_model(model, train_iter, val_iter):
    """Train with early stopping and learning-rate scheduling."""
    cw = compute_weights(train_iter)

    cb_list = [
        callbacks.EarlyStopping(
            monitor='val_loss', patience=4,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.3,
            patience=2, min_lr=1e-7, verbose=1
        ),
        callbacks.ModelCheckpoint(
            os.path.join(RESULTS_DIR, 'checkpoint.keras'),
            monitor='val_loss', save_best_only=True, verbose=0
        )
    ]

    print("\n>> Starting training ...\n")

    history = model.fit(
        train_iter,
        epochs=MAX_EPOCHS,
        validation_data=val_iter,
        class_weight=cw,
        callbacks=cb_list
    )
    return history


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------
def plot_curves(history):
    """Plot accuracy and loss curves side by side."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5))

    a1.plot(epochs_range, acc, 'o-', label='Train', color='#1f77b4')
    a1.plot(epochs_range, val_acc, 's-', label='Validation', color='#ff7f0e')
    a1.set_title('Accuracy per Epoch', fontweight='bold')
    a1.set_xlabel('Epoch')
    a1.set_ylabel('Accuracy')
    a1.legend()
    a1.grid(alpha=0.25)

    a2.plot(epochs_range, loss, 'o-', label='Train', color='#1f77b4')
    a2.plot(epochs_range, val_loss, 's-', label='Validation', color='#ff7f0e')
    a2.set_title('Loss per Epoch', fontweight='bold')
    a2.set_xlabel('Epoch')
    a2.set_ylabel('Loss')
    a2.legend()
    a2.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'learning_curves.png'), dpi=150)
    plt.close()
    print(">> Saved learning_curves.png")


def plot_roc(true_labels, pred_probs):
    """Plot the ROC curve and report AUC."""
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve', fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'roc_curve.png'), dpi=150)
    plt.close()
    print(f">> Saved roc_curve.png  (AUC = {roc_auc:.4f})")
    return roc_auc


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def run_evaluation(model, test_iter):
    """Full evaluation: loss, accuracy, classification report, confusion matrix, ROC."""
    print("\n" + "#" * 55)
    print("#   EVALUATION ON TEST SET")
    print("#" * 55)

    loss, acc = model.evaluate(test_iter, verbose=0)
    print(f"\n  Test loss:     {loss:.4f}")
    print(f"  Test accuracy: {acc:.4f}  ({acc * 100:.2f} %)")

    probs = model.predict(test_iter, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)
    truth = test_iter.classes
    names = list(test_iter.class_indices.keys())

    report_str = classification_report(truth, preds, target_names=names)
    print(f"\n{report_str}")

    # Confusion matrix
    cm = confusion_matrix(truth, preds)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=names, yticklabels=names, linewidths=0.5)
    plt.title('Confusion Matrix (Test Set)', fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(">> Saved confusion_matrix.png")

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1 = 2 * precision * sensitivity / (precision + sensitivity)

    print(f"\n  Sensitivity (Pneumonia recall): {sensitivity:.4f}")
    print(f"  Specificity (Normal recall):   {specificity:.4f}")
    print(f"  Precision:                     {precision:.4f}")
    print(f"  F1 Score:                      {f1:.4f}")

    # ROC
    roc_auc = plot_roc(truth, probs)

    # Save report
    with open(os.path.join(RESULTS_DIR, 'evaluation_report.txt'), 'w') as fh:
        fh.write("Chest X-Ray Pneumonia Detection — Evaluation Report\n")
        fh.write("=" * 52 + "\n\n")
        fh.write(f"Test Loss:     {loss:.4f}\n")
        fh.write(f"Test Accuracy: {acc:.4f}\n")
        fh.write(f"AUC:           {roc_auc:.4f}\n")
        fh.write(f"Sensitivity:   {sensitivity:.4f}\n")
        fh.write(f"Specificity:   {specificity:.4f}\n")
        fh.write(f"Precision:     {precision:.4f}\n")
        fh.write(f"F1 Score:      {f1:.4f}\n\n")
        fh.write(report_str)
    print(">> Saved evaluation_report.txt")

    return loss, acc


def demonstrate_predictions(model, test_dir):
    """Show predictions on a few random test images."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    for row, label in enumerate(['NORMAL', 'PNEUMONIA']):
        folder = os.path.join(test_dir, label)
        picked = np.random.choice(os.listdir(folder), size=3, replace=False)
        for col, fname in enumerate(picked):
            fpath = os.path.join(folder, fname)
            img = keras.preprocessing.image.load_img(fpath, target_size=TARGET_SIZE)
            arr = keras.preprocessing.image.img_to_array(img) / 255.0
            prob = model.predict(np.expand_dims(arr, 0), verbose=0)[0][0]

            pred_label = 'PNEUMONIA' if prob >= 0.5 else 'NORMAL'
            conf = prob if prob >= 0.5 else 1 - prob
            is_right = pred_label == label
            colour = '#27ae60' if is_right else '#c0392b'

            axes[row, col].imshow(img, cmap='bone')
            axes[row, col].set_title(
                f'True: {label}\nPred: {pred_label} ({conf:.1%})',
                color=colour, fontsize=10, fontweight='bold'
            )
            axes[row, col].axis('off')

    plt.suptitle('Sample Predictions  (green = correct, red = wrong)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'sample_predictions.png'), dpi=150)
    plt.close()
    print(">> Saved sample_predictions.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 55)
    print("  CHEST X-RAY PNEUMONIA CLASSIFICATION")
    print("  TensorFlow / Keras CNN Pipeline")
    print("=" * 55)

    dataset_summary()

    show_grid(os.path.join(PATHS['train'], 'NORMAL'), 'Normal X-Rays')
    show_grid(os.path.join(PATHS['train'], 'PNEUMONIA'), 'Pneumonia X-Rays')

    train_it, val_it, test_it = build_data_pipeline()

    net = create_classifier()
    net = compile_and_summarise(net)

    hist = fit_model(net, train_it, val_it)
    plot_curves(hist)

    loss, acc = run_evaluation(net, test_it)

    demonstrate_predictions(net, PATHS['test'])

    net.save(os.path.join(RESULTS_DIR, 'xray_classifier.keras'))
    print(f"\n>> Model saved to results/xray_classifier.keras")

    print("\n" + "=" * 55)
    print(f"  DONE — Test accuracy {acc:.2%}")
    print("=" * 55 + "\n")


if __name__ == '__main__':
    main()
