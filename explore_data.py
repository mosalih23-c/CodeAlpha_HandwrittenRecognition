
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# ======================================
# 1. Load MNIST dataset
# ======================================

print("Loading MNIST dataset...")

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("\n========== DATASET INFORMATION ==========")
print("Training images:", X_train.shape)
print("Training labels:", y_train.shape)
print("Testing images:", X_test.shape)
print("Testing labels:", y_test.shape)

# ======================================
# 2. Explore pixel values
# ======================================

print("\n========== PIXEL INFORMATION ==========")
print("Image dimensions:", X_train[0].shape)
print("Minimum pixel value:", X_train.min())
print("Maximum pixel value:", X_train.max())
print("Data type:", X_train.dtype)

# ======================================
# 3. Check class distribution
# ======================================

print("\n========== CLASS DISTRIBUTION ==========")

digits, counts = np.unique(y_train, return_counts=True)

for digit, count in zip(digits, counts):
    print(f"Digit {digit}: {count} images")

# ======================================
# 4. Visualize sample images
# ======================================

fig, axes = plt.subplots(3, 5, figsize=(12, 7))

for i, ax in enumerate(axes.flat):
    ax.imshow(X_train[i], cmap="gray")
    ax.set_title(f"Label: {y_train[i]}")
    ax.axis("off")

plt.suptitle(
    "MNIST - Handwritten Digit Samples",
    fontsize=16
)

plt.tight_layout()
plt.savefig("mnist_samples.png", dpi=300)
plt.show()

# ======================================
# 5. Visualize class distribution
# ======================================

plt.figure(figsize=(10, 5))

plt.bar(digits, counts)

plt.title("MNIST Training Class Distribution")
plt.xlabel("Digit")
plt.ylabel("Number of Images")
plt.xticks(range(10))
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("class_distribution.png", dpi=300)
plt.show()

print("\nDataset exploration completed successfully!")
