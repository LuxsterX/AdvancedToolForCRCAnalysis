"""
Moduł app.py

Główna aplikacja Flask do obsługi API dla analizy obrazów medycznych.
"""

from flask import Flask
from flask_cors import CORS
from routes import analyze

app = Flask(__name__)
CORS(app)


app.register_blueprint(analyze)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
