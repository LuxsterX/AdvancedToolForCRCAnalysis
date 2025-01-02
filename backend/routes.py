"""
Moduł routes.py

Zawiera definicję endpointu API do analizy obrazów z wykorzystaniem Flask.
"""

import tensorflow as tf
from flask import Blueprint, request, jsonify, Response
from utils import sliding_window
from models import model, TUM_CLASS_INDEX, generate_grad_cam
from PIL import Image
import numpy as np
import base64
import io
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

analyze = Blueprint('analyze', __name__)

@analyze.route("/analyze", methods=["POST"])
def analyze_endpoint():
    """
        Endpoint analizy obrazów.

        Funkcjonalność:
        - Przyjmuje obraz przesłany przez użytkownika.
        - Generuje mapę ciepła Grad-CAM dla zmiany nowotworowej.
        - Oblicza maksymalne i średnie prawdopodobieństwo zmiany nowotworowej.
        - Oblicza procent obszaru gorącego (potencjalnie nowotworowego).
        - Zwraca wyniki w formacie JSON, w tym obraz z nałożoną mapą ciepła.

        Zwraca:
        - JSON zawierający wyniki analizy lub komunikat o błędzie.
        """
    try:
        if not model:
            return jsonify({"error": "Model not loaded. Check server logs."}), 500

        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["image"]
        image = Image.open(file.stream).convert("RGB")
        image = np.array(image, dtype=np.float32)

        h, w, _ = image.shape
        combined_heatmap = np.zeros((h, w), dtype=np.float32)
        counts = np.zeros((h, w), dtype=np.int32)

        for x, y, patch in sliding_window(image, 128, 16):
            patch = patch / 255.0
            heatmap = generate_grad_cam(patch, model, TUM_CLASS_INDEX)
            heatmap_resized = tf.image.resize(heatmap, (128, 128)).numpy().squeeze()

            combined_heatmap[y:y + 128, x:x + 128] += heatmap_resized
            counts[y:y + 128, x:x + 128] += 1

        combined_heatmap = combined_heatmap / np.maximum(counts, 1)
        combined_heatmap = gaussian_filter(combined_heatmap, sigma=3)
        combined_heatmap = np.clip(combined_heatmap, 0, 1)

        cmap = plt.get_cmap("jet")
        norm = Normalize(vmin=0, vmax=1)
        heatmap_colored = cmap(norm(combined_heatmap))[:, :, :3]
        overlay = (0.6 * image / 255.0 + 0.4 * heatmap_colored)
        overlay = np.clip(overlay, 0, 1)

        tumor_probability_max = float(np.max(combined_heatmap))
        tumor_probability_mean = float(np.mean(combined_heatmap))
        threshold = 0.5
        tumor_hot_area = float(np.sum(combined_heatmap > threshold) / combined_heatmap.size)

        overlay_image = Image.fromarray((overlay * 255).astype(np.uint8))
        buffered = io.BytesIO()
        overlay_image.save(buffered, format="PNG")
        overlay_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return jsonify({
            "heatmap_overlay": overlay_base64,
            "tumor_probability_max": tumor_probability_max,
            "tumor_probability_mean": tumor_probability_mean,
            "tumor_hot_area": tumor_hot_area
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
