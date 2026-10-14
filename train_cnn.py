
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import seaborn as sns

# =====================================
# 1. Configuration
# =====================================

SEED = 42
EPOCHS = 15
BATCH_SIZE = 128

np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# =====================================
# 2. Load MNIST
# =====================================

print("Loading MNIST dataset...")

(X_train_full, y_train_full), (X_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# Normalize pixel values from 0-255 to 0-1
X_train_full = X_train_full.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# CNN input shape: samples, height, width, channels
X_train_full = X_train_full[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Create validation set from training data
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.10,
    random_state=SEED,
    stratify=y_train_full
)

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

# =====================================
# 3. Build CNN
# =====================================

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(
        32, kernel_size=3,
        padding="same",
        activation="relu"
    ),
    layers.MaxPooling2D(pool_size=2),

    layers.Conv2D(
        64, kernel_size=3,
        padding="same",
        activation="relu"
    ),
    layers.MaxPooling2D(pool_size=2),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.30),

    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =====================================
# 4. Training callbacks
# =====================================

model_path = os.path.join(
    MODEL_DIR,
    "mnist_cnn.keras"
)

training_callbacks = [
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

# =====================================
# 5. Train
# =====================================

print("\nStarting CNN training...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=training_callbacks,
    verbose=1
)

# Load the best model checkpoint
model = tf.keras.models.load_model(model_path)

# =====================================
# 6. Evaluate on held-out test set
# =====================================

print("\nEvaluating on test set...")

probabilities = model.predict(X_test, batch_size=BATCH_SIZE)
y_pred = np.argmax(probabilities, axis=1)

test_accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {test_accuracy:.4f}")
print("\nClassification Report:")
report = classification_report(
    y_test,
    y_pred,
    digits=4,
    zero_division=0
)
print(report)

with open(
    os.path.join(REPORT_DIR, "classification_report.txt"),
    "w",
    encoding="utf-8"
) as file:
    file.write(f"Test Accuracy: {test_accuracy:.4f}\n\n")
    file.write(report)

with open(
    os.path.join(REPORT_DIR, "metrics.json"),
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        {"test_accuracy": float(test_accuracy)},
        file,
        indent=4
    )

# =====================================
# 7. Confusion matrix
# =====================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(9, 7))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=range(10),
    yticklabels=range(10)
)
plt.title("MNIST CNN - Confusion Matrix")
plt.xlabel("Predicted Digit")
plt.ylabel("Actual Digit")
plt.tight_layout()

plt.savefig(
    os.path.join(REPORT_DIR, "confusion_matrix.png"),
    dpi=300
)
plt.close()

# =====================================
# 8. Training curves
# =====================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(history.history["accuracy"], label="Training")
axes[0].plot(history.history["val_accuracy"], label="Validation")
axes[0].set_title("Model Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history["loss"], label="Training")
axes[1].plot(history.history["val_loss"], label="Validation")
axes[1].set_title("Model Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(REPORT_DIR, "training_curves.png"),
    dpi=300
)
plt.close()

print("\nCNN training completed successfully.")
print("Saved model:", model_path)
print("Reports directory:", REPORT_DIR)
