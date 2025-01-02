"""
Moduł ModelTrainer.py

Funkcje:
- Podział danych na zestawy treningowe i walidacyjne.
- Przygotowanie generatorów danych.
- Tworzenie i trenowanie modelu.
- Wizualizacja wyników treningu.
"""

import os
import shutil
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def split_data(base_dir, output_dir, test_size=0.2):
    """
    Dzieli dane z folderu na zestawy treningowe i walidacyjne.

    Parametry:
    - base_dir (str): Ścieżka do folderu z oryginalnymi danymi (NCT-CRC-HE-100K).
    - output_dir (str): Ścieżka do folderu wyjściowego z podzielonymi danymi.
    - test_size (float): Proporcja danych walidacyjnych (np. 0.2 dla 20%).
    """
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for class_name in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_name)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            train_images, val_images = train_test_split(images, test_size=test_size, random_state=42)

            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

            for image in train_images:
                shutil.copy(os.path.join(class_path, image), os.path.join(train_dir, class_name, image))
            for image in val_images:
                shutil.copy(os.path.join(class_path, image), os.path.join(val_dir, class_name, image))

    print(f"Dane zostały podzielone i zapisane w folderze: {output_dir}")

BASE_DIR = "D:/data/NCT-CRC-HE-100K"
OUTPUT_DIR = "D:/data/NCT-CRC-HE-100K-split"

split_data(BASE_DIR, OUTPUT_DIR)

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
NUM_CLASSES = 9

# Generator danych treningowych
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    channel_shift_range=20.0,
    horizontal_flip=True
)

# Generator danych walidacyjnych
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(OUTPUT_DIR, "train"),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(OUTPUT_DIR, "val"),
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Model bazowy ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(*IMAGE_SIZE, 3))

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = models.Model(inputs=base_model.input, outputs=outputs)

for layer in base_model.layers[-10:]:
    layer.trainable = True

# Kompilacja modelu z optymalizatorem Adam, stratą kategorycznej entropii krzyżowej i dokładnością jako metryką.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpoint = ModelCheckpoint("nct_crc_model.keras", save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Trening modelu z generatorami danych i wczesnym zatrzymywaniem.
history = model.fit(
    train_generator,
    validation_data=val_generator,
    steps_per_epoch=2500,
    validation_steps=625,
    epochs=30,
    verbose=1,
    callbacks=[checkpoint, early_stopping]
)

def plot_training_history(history):
    """
        Tworzy wykresy przedstawiające historię treningu:
        - Dokładność dla danych treningowych i walidacyjnych.
        - Straty dla danych treningowych i walidacyjnych.

        Parametry:
        - history: Historia treningu zwrócona przez funkcję model.fit.
        """
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    plt.show()

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Model Loss')
    plt.show()

plot_training_history(history)

def grad_cam(model, img_array, class_index, layer_name):
    """
        Generuje mapę ciepła Grad-CAM dla podanego obrazu.

        Parametry:
        - model: Model TensorFlow/Keras.
        - img_array (numpy.ndarray): Obraz wejściowy.
        - class_index (int): Indeks klasy.
        - layer_name (str): Nazwa warstwy modelu do analizy.

        Zwraca:
        - heatmap (numpy.ndarray): Mapa ciepła Grad-CAM.
        """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    return heatmap.numpy()

def display_grad_cam(image_path, model, class_index, layer_name="conv5_block3_out"):
    """
        Wyświetla obraz z nałożoną mapą ciepła Grad-CAM.

        Parametry:
        - image_path (str): Ścieżka do obrazu.
        - model: Model TensorFlow/Keras.
        - class_index (int): Indeks klasy.
        - layer_name (str): Nazwa warstwy modelu do analizy.
        """
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    heatmap = grad_cam(model, img_array, class_index, layer_name)

    plt.imshow(img)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.show()
