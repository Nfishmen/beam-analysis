"""
Flask web server for Beam Analysis System.
Provides an upload UI and a JSON API.

Usage:
    python app.py
     open http://127.0.0.1:5000
"""

import os
import sys
from pathlib import Path

# ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from pipeline import BeamAnalysisPipeline

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
RESULTS_FOLDER = BASE_DIR / "static" / "results"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

# ---------------------------------------------------------------------------
# Global pipeline instance (lazy init)
# ---------------------------------------------------------------------------
pipeline: "BeamAnalysisPipeline | None" = None


def get_pipeline() -> BeamAnalysisPipeline:
    global pipeline
    if pipeline is None:
        pipeline = BeamAnalysisPipeline()
    return pipeline


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    """Serve the main upload page."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """
    Receive an image, run the analysis pipeline, return JSON results.

    Response:
        {
            success: bool,
            mode: "yolo" | "demo",
            beam_model: {...},
            results: {...},
            charts: {report, stress, detection_overlay?},
            detections: [...]
        }
    """
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"success": False, "error": f"Unsupported format: .{ext}"}), 400

    # save uploaded file
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    # run pipeline
    try:
        pipe = get_pipeline()
        result = pipe.run(save_path, output_dir=str(RESULTS_FOLDER))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Analysis failed: {str(e)}"}), 500

    # include the original image URL
    result["original_image"] = f"/static/uploads/{filename}"

    return jsonify(result)


@app.route("/static/<path:subpath>")
def static_files(subpath: str):
    """Serve generated results and uploaded images."""
    return send_from_directory(BASE_DIR / "static", subpath)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 56)
    print("    [Beam] Structure Analysis System")
    print("  Opening browser at http://127.0.0.1:5000 ...")
    print("=" * 56)
    import webbrowser, threading
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(host="0.0.0.0", port=5000, debug=False)
