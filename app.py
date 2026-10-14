
import streamlit as st
import numpy as np
import tensorflow as tf

from PIL import Image, ImageOps, ImageFilter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "mnist_cnn.keras"

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="wide"
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def otsu_threshold(image_array):
    """Calculate Otsu threshold without requiring OpenCV."""
    histogram = np.bincount(
        image_array.ravel(), minlength=256
    ).astype(np.float64)

    total = image_array.size
    total_sum = np.dot(np.arange(256), histogram)

    background_weight = 0
    background_sum = 0
    best_variance = -1
    threshold = 127

    for value in range(256):
        background_weight += histogram[value]

        if background_weight == 0:
            continue

        foreground_weight = total - background_weight

        if foreground_weight == 0:
            break

        background_sum += value * histogram[value]

        background_mean = background_sum / background_weight
        foreground_mean = (
            total_sum - background_sum
        ) / foreground_weight

        variance = (
            background_weight
            * foreground_weight
            * (background_mean - foreground_mean) ** 2
        )

        if variance > best_variance:
            best_variance = variance
            threshold = value

    return threshold



def preprocess_image(image):
    # 1. Convert image to RGB and HSV.
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8)

    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # 2. Detect blue / cyan ink.
    # PIL HSV hue ranges from 0 to 255.
    blue_ink = (
        (hue >= 115) &
        (hue <= 195) &
        (saturation > 45) &
        (value < 245)
    )

    # Also detect dark ink, such as black pen.
    gray = np.asarray(
        ImageOps.grayscale(image).filter(
            ImageFilter.MedianFilter(size=3)
        ),
        dtype=np.uint8
    )

    dark_ink = gray < 95

    # Combine color and dark-ink masks.
    mask = blue_ink | dark_ink

    # 3. Remove rows likely to be notebook lines.
    # Long horizontal lines usually occupy many pixels in a row.
    height, width = mask.shape
    row_counts = mask.sum(axis=1)

    line_rows = row_counts > max(20, int(width * 0.25))

    # Remove detected line rows and a small margin around them.
    rows_to_remove = line_rows.copy()

    for shift in (1, 2):
        rows_to_remove[shift:] |= line_rows[:-shift]
        rows_to_remove[:-shift] |= line_rows[shift:]

    mask[rows_to_remove, :] = False

    # 4. Find the remaining foreground.
    ys, xs = np.where(mask)

    if len(xs) < 10:
        raise ValueError(
            "Could not isolate the digit. Try a clearer photo "
            "with stronger contrast."
        )

    # 5. Crop around the detected digit.
    x_min, x_max = xs.min(), xs.max() + 1
    y_min, y_max = ys.min(), ys.max() + 1

    cropped_mask = mask[y_min:y_max, x_min:x_max]

    # Convert foreground to white and background to black.
    digit = Image.fromarray(
        (cropped_mask.astype(np.uint8) * 255),
        mode="L"
    )

    # 6. Resize proportionally to fit within 20x20.
    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # 7. Center on a 28x28 black canvas.
    canvas = Image.new("L", (28, 28), color=0)

    x_offset = (28 - digit.width) // 2
    y_offset = (28 - digit.height) // 2

    canvas.paste(digit, (x_offset, y_offset))

    # 8. Prepare input for the CNN.
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    array = array[np.newaxis, ..., np.newaxis]

    return canvas, array



st.title("🔢 Handwritten Digit Recognition")

st.write(
    "Upload an image containing one handwritten digit. "
    "The CNN model predicts a digit from 0 to 9."
)

st.info(
    "Use one dark, clearly written digit. The app attempts to remove "
    "background noise and crop the digit automatically."
)

uploaded_file = st.file_uploader(
    "Upload a digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    try:
        original_image = Image.open(uploaded_file).convert("RGB")

        processed_image, input_array = preprocess_image(
            original_image
        )

        model = load_model()
        probabilities = model.predict(input_array, verbose=0)[0]

        predicted_digit = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_digit]) * 100

        left, right = st.columns(2)

        with left:
            st.subheader("Original Image")
            st.image(
                original_image,
                caption="Uploaded photo",
                use_container_width=True
            )

        with right:
            st.subheader("Processed Model Input")
            st.image(
                processed_image,
                caption="28 × 28 pixels — white digit on black background",
                clamp=True,
                width=224
            )

        st.divider()

        st.success(f"Predicted Digit: {predicted_digit}")
        st.metric("Model Confidence", f"{confidence:.2f}%")

        st.subheader("Prediction Probabilities")

        chart_data = {
            "Digit": [str(i) for i in range(10)],
            "Probability": probabilities
        }

        st.bar_chart(
            chart_data,
            x="Digit",
            y="Probability"
        )

        st.caption(
            "Confidence is the model's predicted probability, "
            "not a guarantee that the prediction is correct."
        )

        st.warning(
            "Check the processed image above. If the digit is missing, "
            "too small, or the background is white, the prediction "
            "may be unreliable."
        )

    except Exception as error:
        st.error(f"Could not process this image: {error}")
