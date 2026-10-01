import io
import base64
import numpy as np
from PIL import Image
import matplotlib
import tensorflow as tf
from tensorflow import keras


def make_gradcam_heatmap(img_array: np.ndarray, model: keras.Model, last_conv_layer_name: str = "conv_final", pred_index: int = None) -> np.ndarray:
    """
    Computes Grad-CAM heatmap for a given input image array and model.
    """
    # 1. Locate the target convolutional layer
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except ValueError:
        # Fallback: find the last Conv2D layer in the model
        conv_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.Conv2D)]
        if not conv_layers:
            raise ValueError("No Conv2D layer found in the model for Grad-CAM.")
        last_conv_layer = conv_layers[-1]

    # 2. Construct gradient model mapping inputs to (last conv layer output, model output)
    grad_model = keras.Model(
        inputs=[model.inputs],
        outputs=[last_conv_layer.output, model.output]
    )

    # 3. Compute gradients with GradientTape
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradient of the winning class with respect to the output feature map
    grads = tape.gradient(class_channel, conv_outputs)

    # Vector where each entry is the mean intensity of the gradient over a specific feature-map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight each channel in the feature map by how important it is for the class
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU: only positive contributions matter
    heatmap = tf.maximum(heatmap, 0.0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def overlay_heatmap_on_image(
    original_img: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap_name: str = "jet"
) -> tuple[Image.Image, Image.Image, dict]:
    """
    Overlays a Grad-CAM heatmap onto the original PIL image.
    Returns:
      (blended_image, pure_heatmap_image, localization_metadata)
    """
    orig_w, orig_h = original_img.size
    orig_rgb = original_img.convert("RGB")
    orig_np = np.array(orig_rgb)

    # 1. Resize heatmap to original image size
    heatmap_pil = Image.fromarray(np.uint8(255 * heatmap)).resize((orig_w, orig_h), Image.Resampling.BILINEAR)
    heatmap_resized = np.array(heatmap_pil) / 255.0

    # 2. Use colormap to colorize heatmap (matplotlib 3.9+ compatible)
    try:
        colormap = matplotlib.colormaps[colormap_name]
    except AttributeError:
        # Fallback for older matplotlib
        colormap = matplotlib.cm.get_cmap(colormap_name)  # type: ignore
    colored_heatmap = colormap(heatmap_resized)[:, :, :3]  # (H, W, 3) in [0, 1]
    colored_heatmap_uint8 = np.uint8(255 * colored_heatmap)
    heatmap_image = Image.fromarray(colored_heatmap_uint8)

    # 3. Blend original image with colored heatmap
    blended = np.uint8(orig_np * (1.0 - alpha) + colored_heatmap_uint8 * alpha)
    blended_image = Image.fromarray(blended)

    # 4. Compute localization metrics (tumor center of mass & bounding region)
    # Threshold heatmap at top 30% intensity
    threshold = 0.4
    mask = heatmap_resized > threshold
    if np.any(mask):
        y_indices, x_indices = np.where(mask)
        center_x = int(np.mean(x_indices))
        center_y = int(np.mean(y_indices))
        min_x, max_x = int(np.min(x_indices)), int(np.max(x_indices))
        min_y, max_y = int(np.min(y_indices)), int(np.max(y_indices))
        area_pct = round(float(np.sum(mask)) / (orig_w * orig_h) * 100, 2)
    else:
        center_x, center_y = orig_w // 2, orig_h // 2
        min_x, max_x = 0, 0
        min_y, max_y = 0, 0
        area_pct = 0.0

    localization = {
        "center": {"x": center_x, "y": center_y},
        "bbox": {"x1": min_x, "y1": min_y, "x2": max_x, "y2": max_y},
        "lesion_coverage_pct": area_pct,
    }

    return blended_image, heatmap_image, localization


def pil_to_base64(img: Image.Image, format: str = "JPEG") -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
