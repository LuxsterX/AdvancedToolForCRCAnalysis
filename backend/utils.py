"""
Moduł utils.py

Zawiera funkcje pomocnicze, takie jak generowanie okien przesuwnych dla obrazów.
"""

def sliding_window(image, window_size, step_size):
    """
        Generator przesuwnego okna dla obrazów.

        Parametry:
        - image (numpy.ndarray): Obraz wejściowy.
        - window_size (int): Rozmiar okna (np. 128x128 pikseli).
        - step_size (int): Wielkość kroku przesunięcia okna.

        Generuje:
        - Położenie okna (x, y) oraz wycinek obrazu odpowiadający oknu.
        """
    h, w, _ = image.shape
    for y in range(0, h - window_size + 1, step_size):
        for x in range(0, w - window_size + 1, step_size):
            yield x, y, image[y:y + window_size, x:x + window_size]

    if h % step_size != 0:
        for x in range(0, w - window_size + 1, step_size):
            yield x, h - window_size, image[h - window_size:h, x:x + window_size]

    if w % step_size != 0:
        for y in range(0, h - window_size + 1, step_size):
            yield w - window_size, y, image[y:y + window_size, w - window_size:w]

    if h % step_size != 0 and w % step_size != 0:
        yield w - window_size, h - window_size, image[h - window_size:h, w - window_size:w]
