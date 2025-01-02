"""
Moduł models.py

Zawiera kod odpowiedzialny za ładowanie modelu oraz generowanie map ciepła Grad-CAM.
"""

import os
import tensorflow as tf

MODEL_PATH = "nct_crc_model.keras"
CLASS_LABELS = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
TUM_CLASS_INDEX = CLASS_LABELS.index("TUM")

if os.path.exists(MODEL_PATH):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
else:
    print(f"Model does not exist at path: {MODEL_PATH}")
    model = None

def generate_grad_cam(image, model, class_index, layer_name="conv5_block3_out"):
    """
        Generuje mapę ciepła Grad-CAM dla podanego obrazu.

        Parametry:
        - image (numpy.ndarray): Obraz wejściowy.
        - model: Model TensorFlow/Keras.
        - class_index (int): Indeks klasy do analizy.
        - layer_name (str): Nazwa warstwy modelu użytej do obliczeń.

        Zwraca:
        - heatmap (tensorflow.Tensor): Mapa ciepła Grad-CAM w formacie TensorFlow.
        """
    grad_model = tf.keras.models.Model(
        [model.input], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tf.expand_dims(image, axis=0))
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    return tf.expand_dims(heatmap, axis=-1)
