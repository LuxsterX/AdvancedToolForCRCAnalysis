/**
 * Główna aplikacja React do analizy obrazów medycznych.
 *
 * Funkcje:
 * - Przesyłanie obrazów.
 * - Analiza z wykorzystaniem backendu Flask.
 * - Wyświetlanie wyników analizy.
 */

import React, { useState } from "react";
import axios from "axios";
import "./App.css";


/**
 * Komponent główny aplikacji.
 *
 * Stan aplikacji:
 * - image: Przesłany obraz przez użytkownika.
 * - processedImage: Obraz z nałożoną mapą ciepła.
 * - tumorProbabilityMax: Maksymalne prawdopodobieństwo wystąpienia zmiany nowotworowej.
 * - tumorProbabilityMean: Średnie prawdopodobieństwo zmiany nowotworowej.
 * - tumorHotArea: Procent obszaru wykrytego jako gorący (potencjalnie nowotworowy).
 */
function App() {
  const [image, setImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);
  const [tumorProbabilityMax, setTumorProbabilityMax] = useState(null);
  const [tumorProbabilityMean, setTumorProbabilityMean] = useState(null);
  const [tumorHotArea, setTumorHotArea] = useState(null);

  // Funkcja do resetowania stanu aplikacji

  /**
   * Resetuje stan aplikacji do wartości początkowych.
   */
  const resetApp = () => {
    setImage(null);
    setProcessedImage(null);
    setTumorProbabilityMax(null);
    setTumorProbabilityMean(null);
    setTumorHotArea(null);
  };


  /**
   * Obsługuje przesyłanie obrazu przez użytkownika.
   *
   * @param {Object} e - Event przesyłania pliku.
   */
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    resetApp(); // Reset aplikacji przy przesyłaniu nowego obrazu
    setImage(file);
  };


  /**
   * Wysyła obraz do backendu w celu analizy i odbiera wyniki.
   *
   * - Wysyła żądanie POST z obrazem.
   * - Obsługuje zdarzenia SSE dla informacji o postępie.
   * - Aktualizuje stan aplikacji na podstawie wyników analizy.
   */
  const sendRequest = async () => {
    if (!image) {
      alert("Please upload an image first!");
      return;
    }

    const formData = new FormData();
    formData.append("image", image);

    try {
      // Otwórz strumień EventSource
      const eventSource = new EventSource("http://localhost:5000/analyze");

      eventSource.onmessage = (event) => {
        const progressValue = parseInt(event.data, 10);
        console.log("Received progress:", progressValue); // Debugowanie

        if (progressValue >= 100) {
          eventSource.close(); // Zamknij SSE po zakończeniu analizy
        }
      };

      eventSource.onerror = () => {
        console.error("Error with SSE connection.");
        eventSource.close();
      };

      // Odbierz dane analizy po zakończeniu
      const response = await axios.post("http://localhost:5000/analyze", formData);
      const data = response.data;

      if (data.heatmap_overlay) {
        setProcessedImage(`data:image/png;base64,${data.heatmap_overlay}`);
      }

      setTumorProbabilityMax(data.tumor_probability_max || null);
      setTumorProbabilityMean(data.tumor_probability_mean || null);
      setTumorHotArea(data.tumor_hot_area || null);
    } catch (error) {
      console.error(error);
      alert("Error processing the request.");
    }
  };

  return (
      <div className="app-container">
        <div className="upload-container">
          <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              style={{ display: "none" }}
              id="file-upload"
          />
          <label htmlFor="file-upload" className="upload-button">
            ADD IMAGE
          </label>
        </div>

        <div className="image-container">
          <div className="original-container">
            <h3>Original Image</h3>
            <div className="placeholder">
              {image ? <img src={URL.createObjectURL(image)} alt="Uploaded" className="uploaded-image" /> : "+"}
            </div>
          </div>
          <div className="processed-container">
            <h3>Processed Image</h3>
            <div className="processed-placeholder">
              {processedImage ? (
                  <img src={processedImage} alt="Heatmap" className="uploaded-image" />
              ) : (
                  "Processed heatmap will appear here"
              )}
            </div>
          </div>
        </div>

        <div className="controls-container">
          <button className="control-button" onClick={sendRequest} disabled={!image}>
            ANALYZE
          </button>
        </div>

        <div className="results-container">
          <h3>Results</h3>
          {tumorProbabilityMax !== null && tumorProbabilityMean !== null && tumorHotArea !== null ? (
              <div>
                <div className="result-text">Max Probability: {(tumorProbabilityMax * 100).toFixed(2)}%</div>
                <div className="result-text">Mean Probability: {(tumorProbabilityMean * 100).toFixed(2)}%</div>
                <div className="result-text">Hot Area Percentage: {(tumorHotArea * 100).toFixed(2)}%</div>
              </div>
          ) : (
              <div className="result-text">No results yet</div>
          )}
        </div>
      </div>
  );
}

export default App;
