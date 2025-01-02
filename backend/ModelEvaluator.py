"""
Moduł ModelEvaluator.py

Klasa ModelEvaluator:
- Wczytywanie modelu.
- Generowanie danych testowych.
- Analiza obrazów przy pomocy Grad-CAM.
- Obliczanie metryk i generowanie wizualizacji.

Użycie:
evaluator = ModelEvaluator(model_path, test_images_dir, class_labels)
evaluator.run_all()
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_fscore_support
import seaborn as sns
from tensorflow.keras.models import load_model
import random


"""
Klasa ModelEvaluator

Atrybuty:
- model_path: Ścieżka do zapisanego modelu.
- test_images_dir: Folder z obrazami testowymi.
- class_labels: Lista nazw klas.
- image_size: Rozmiar obrazów wejściowych.

Metody:
- load_model(): Ładuje model z pliku.
- generate_test_data(): Generuje dane testowe z folderu.
- grad_cam(): Generuje mapy ciepła Grad-CAM.
- test_model_with_visualizations_side_by_side(): Tworzy porównania wizualne.
- evaluate_metrics(): Oblicza metryki i generuje raporty.
- plot_training_history(): Tworzy wykresy historii treningu.
- run_all(): Uruchamia wszystkie kroki.
"""
class ModelEvaluator:

    """
    Inicjalizuje klasę ModelEvaluator.

    Parametry:
    - model_path (str): Ścieżka do modelu.
    - test_images_dir (str): Ścieżka do folderu z danymi testowymi.
    - class_labels (list): Lista etykiet klas.
    - image_size (tuple): Rozmiar obrazu (domyślnie (128, 128)).
    """
    def __init__(self, model_path, test_images_dir, class_labels, image_size=(128, 128)):
        self.model_path = model_path
        self.test_images_dir = test_images_dir
        self.class_labels = class_labels
        self.image_size = image_size
        self.model = None
        self.test_generator = None


    """
    Ładuje zapisany model z podanej ścieżki.
    """
    def load_model(self):
        """Wczytuje model."""
        self.model = load_model(self.model_path)


    """
    Generuje dane testowe z podanego folderu testowego.
    Normalizuje obrazy w zakresie [0, 1].
    """
    def generate_test_data(self):
        """Generuje dane testowe."""
        test_datagen = ImageDataGenerator(rescale=1.0 / 255)
        self.test_generator = test_datagen.flow_from_directory(
            self.test_images_dir,
            target_size=self.image_size,
            batch_size=32,
            class_mode="categorical",
            shuffle=False
        )


    """
    Generuje mapę ciepła Grad-CAM dla danego obrazu.

    Parametry:
    - img_array (numpy.ndarray): Obraz wejściowy.
    - class_index (int): Indeks klasy do analizy.
    - layer_name (str): Nazwa warstwy modelu dla Grad-CAM.

    Zwraca:
    - heatmap (numpy.ndarray): Mapa ciepła Grad-CAM.
    """
    def grad_cam(self, img_array, class_index, layer_name="conv5_block3_out"):
        """Generuje mapę ciepła Grad-CAM dla podanego obrazu."""
        grad_model = tf.keras.models.Model(
            [self.model.inputs], [self.model.get_layer(layer_name).output, self.model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-10)

        return heatmap


    """
    Tworzy porównania wizualne dla obrazów testowych:
    - Wyświetla oryginalne obrazy.
    - Nakłada mapy ciepła Grad-CAM.
    - Zapisuje wyniki do plików SVG.
    """
    def test_model_with_visualizations_side_by_side(self):
        """Tworzy porównania dla 5 losowych obrazów z każdej kategorii z legendą bezpośrednio pod obrazami."""
        n_images = 5
        for class_name in self.class_labels:
            class_path = os.path.join(self.test_images_dir, class_name)
            if os.path.isdir(class_path):
                image_files = random.sample(os.listdir(class_path), min(n_images, len(os.listdir(class_path))))

                fig, axes = plt.subplots(3, n_images, figsize=(20, 12))  # Trzy wiersze: Oryginał, Grad-CAM, Legenda

                for idx, image_name in enumerate(image_files):
                    image_path = os.path.join(class_path, image_name)
                    img = tf.keras.preprocessing.image.load_img(image_path, target_size=self.image_size)
                    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    predicted_class = np.argmax(self.model.predict(img_array))
                    heatmap = self.grad_cam(img_array, predicted_class)

                    heatmap = np.expand_dims(heatmap, axis=-1)
                    heatmap = tf.image.resize(heatmap, self.image_size).numpy().squeeze()

                    # Oryginalny obraz
                    axes[0, idx].imshow(img)
                    axes[0, idx].set_title(f"Oryginał ({class_name})")
                    axes[0, idx].axis("off")

                    # Grad-CAM
                    axes[1, idx].imshow(img)
                    heatmap_overlay = axes[1, idx].imshow(heatmap, cmap="jet", alpha=0.5)
                    axes[1, idx].set_title("Grad-CAM")
                    axes[1, idx].axis("off")


                for ax in axes[2, :]:
                    ax.axis("off")

                cbar = fig.colorbar(
                    heatmap_overlay,
                    ax=axes[2, :],
                    orientation="horizontal",
                    fraction=0.5,
                    pad= -0.5
                )
                cbar.set_label("Aktywacja Grad-CAM", rotation=0, labelpad=10, fontsize=12)

                plt.tight_layout()
                plt.savefig(f"grad_cam_comparison_{class_name}.svg", format="svg")
                plt.close()


    """
    Oblicza i zapisuje metryki ewaluacyjne:
    - Macierz konfuzji.
    - Krzywe ROC i AUC.
    - Raport klasyfikacji w formacie LaTeX.
    """
    def evaluate_metrics(self):
        """Oblicza metryki takie jak F1-score, precision, recall oraz generuje krzywe ROC i AUC."""
        y_pred_probs = self.model.predict(self.test_generator)
        y_true = self.test_generator.classes
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Raport
        report = classification_report(y_true, y_pred, target_names=self.class_labels, output_dict=True)
        latex_table = (
            "\\begin{table}[h!]\n"
            "\\centering\n"
            "\\begin{tabular}{lccc}\n"
            "\\hline\n"
            "Klasa & Precision & Recall & F1-Score \\\\\\ \\hline\n"
        )

        for label in self.class_labels:
            latex_table += (
                f"{label} & {report[label]['precision']:.2f} & "
                f"{report[label]['recall']:.2f} & {report[label]['f1-score']:.2f} \\\\\\ \\n"
            )

        latex_table += (
            "\\hline\n"
            "\\end{tabular}\n"
            "\\caption{Metryki klasyfikacji dla każdej klasy}\n"
            "\\end{table}\n"
        )

        # Zapis pliku
        with open("classification_report.tex", "w", encoding="utf-8") as f:
            f.write(latex_table)

        # Macierz konfuzji
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=self.class_labels, yticklabels=self.class_labels)
        plt.xlabel("Predykcja")
        plt.ylabel("Prawdziwa klasa")
        plt.title("Macierz konfuzji")
        plt.savefig("confusion_matrix.svg", format="svg")
        plt.close()

        # ROC i AUC
        plt.figure(figsize=(10, 8))
        for i, label in enumerate(self.class_labels):
            fpr, tpr, _ = roc_curve(y_true == i, y_pred_probs[:, i])
            auc_score = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{label} (AUC: {auc_score:.2f})")

        plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Krzywe ROC dla każdej klasy")
        plt.legend()
        plt.savefig("roc_curves.svg", format="svg")
        plt.close()


    """
    Tworzy i zapisuje wykresy przedstawiające historię treningu:
    - Dokładność dla danych treningowych i walidacyjnych.
    - Straty dla danych treningowych i walidacyjnych.
    """
    def plot_training_history(self):
        """Tworzy wykresy dokładności i strat na podstawie danych."""

        epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        train_accuracy = [0.92846, 0.96319, 0.97010, 0.97548, 0.97755, 0.98079, 0.98200, 0.98403, 0.98524, 0.98688, 0.98702, 0.98855, 0.98909, 0.98980, 0.99019]
        val_accuracy = [0.92454, 0.92479, 0.92151, 0.92493, 0.92008, 0.97555, 0.97741, 0.97883, 0.90028, 0.98392, 0.97270, 0.94580, 0.94398, 0.98278, 0.97615]
        train_loss = [0.22782, 0.11864, 0.096564, 0.081380, 0.074277, 0.064076, 0.062345, 0.056782, 0.055017, 0.051343, 0.050768, 0.046332, 0.047272, 0.045142, 0.044538]
        val_loss = [0.26157, 0.26060, 0.29721, 0.26555, 0.29949, 0.10092, 0.078459, 0.077471, 0.50583, 0.057487, 0.096803, 0.25697, 0.22184, 0.068935, 0.099029]

        # Wykres dokładności
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_accuracy, label='Dokładność treningu')
        plt.plot(epochs, val_accuracy, label='Dokładność walidacji')
        plt.xlabel('Epoki')
        plt.ylabel('Dokładność')
        plt.title('Dokładność modelu na przestrzeni epok')
        plt.legend()
        plt.grid(True)
        plt.savefig("accuracy_per_epoch.svg", format="svg")
        plt.close()

        # Wykres strat
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, label='Strata treningu')
        plt.plot(epochs, val_loss, label='Strata walidacji')
        plt.xlabel('Epoki')
        plt.ylabel('Strata')
        plt.title('Straty modelu na przestrzeni epok')
        plt.legend()
        plt.grid(True)
        plt.savefig("loss_per_epoch.svg", format="svg")
        plt.close()


    """
    Uruchamia pełny proces ewaluacji:
    - Wczytuje model.
    - Generuje dane testowe.
    - Tworzy wizualizacje Grad-CAM.
    - Oblicza metryki.
    - Tworzy wykresy treningowe.
    """
    def run_all(self):
        """Uruchamia wszystkie kroki analizy."""
        self.load_model()
        self.generate_test_data()
        self.test_model_with_visualizations_side_by_side()
        self.evaluate_metrics()
        self.plot_training_history()

evaluator = ModelEvaluator(
    model_path="nct_crc_model.keras",
    test_images_dir="D:/data/NCT-CRC-HE-100K-split/val",
    class_labels=["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
)

evaluator.run_all()
