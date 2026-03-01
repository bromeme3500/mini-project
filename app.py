"""
Flask Application — Crowd Detection Server
Handles video upload and streams annotated frames via Server-Sent Events.
"""

import os
import uuid
import json
import base64
import time
import threading
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response

from crowd_analyzer import CrowdAnalyzer

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB max upload

# Ensure upload folder exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Global analyzer instance and readiness flag
analyzer = None
models_ready = False
models_loading = False
model_load_error = None


def load_models_background():
    """Load models in a background thread so the server starts fast."""
    global analyzer, models_ready, models_loading, model_load_error
    models_loading = True
    try:
        print("[Server] Loading models in background...")
        analyzer = CrowdAnalyzer(threshold=20)
        models_ready = True
        print("[Server] Models loaded and ready!")
    except Exception as e:
        model_load_error = str(e)
        print(f"[Server] Error loading models: {e}")
    finally:
        models_loading = False


def get_analyzer(threshold=20):
    """Get the analyzer, loading it if not yet loaded."""
    global analyzer, models_ready
    if analyzer is None:
        analyzer = CrowdAnalyzer(threshold=threshold)
        models_ready = True
    else:
        analyzer.set_threshold(threshold)
    return analyzer


@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@app.route("/status")
def status():
    """Return model loading status."""
    return jsonify({
        "ready": models_ready,
        "loading": models_loading,
        "error": model_load_error,
    })


@app.route("/upload", methods=["POST"])
def upload_video():
    """
    Accept a video file upload.
    Returns a JSON object with the video_id for subsequent processing.
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Validate extension
    allowed_extensions = {"mp4", "avi", "mov", "mkv", "wmv", "webm"}
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in allowed_extensions:
        return jsonify({"error": f"Unsupported format. Use: {', '.join(allowed_extensions)}"}), 400

    # Save with unique ID
    video_id = str(uuid.uuid4())
    filename = f"{video_id}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Get video info
    cap = cv2.VideoCapture(filepath)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return jsonify({
        "video_id": video_id,
        "filename": file.filename,
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "ext": ext,
    })


@app.route("/process/<video_id>")
def process_video(video_id):
    """
    Stream processed frames as Server-Sent Events.
    Each event contains: base64-encoded annotated frame, counts, model info.
    """
    threshold = request.args.get("threshold", 20, type=int)
    skip_frames = request.args.get("skip", 2, type=int)  # Process every Nth frame for speed

    def generate():
        global models_ready

        # Find the video file
        video_path = None
        for f in os.listdir(app.config["UPLOAD_FOLDER"]):
            if f.startswith(video_id):
                video_path = os.path.join(app.config["UPLOAD_FOLDER"], f)
                break

        if video_path is None:
            yield f"data: {json.dumps({'error': 'Video not found'})}\n\n"
            return

        # Wait for models to load, sending heartbeat events
        if not models_ready:
            yield f"data: {json.dumps({'status': 'loading_models', 'message': 'Loading AI models (first time takes ~30s)...'})}\n\n"
            timeout = 120  # seconds
            start = time.time()
            while not models_ready and (time.time() - start) < timeout:
                time.sleep(1)
                elapsed = int(time.time() - start)
                yield f"data: {json.dumps({'status': 'loading_models', 'message': f'Loading AI models... ({elapsed}s)'})}\n\n"
            if not models_ready:
                yield f"data: {json.dumps({'error': 'Model loading timed out. Please refresh and try again.'})}\n\n"
                return

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        crowd_analyzer = get_analyzer(threshold)
        crowd_analyzer.reset()

        frame_number = 0
        processed = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1

            # Skip frames for performance
            if frame_number % skip_frames != 0:
                continue

            # Resize large frames for faster processing
            h, w = frame.shape[:2]
            max_dim = 1280
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            # Analyze the frame
            result = crowd_analyzer.analyze_frame(frame)
            processed += 1

            # Encode annotated frame as JPEG base64
            _, buffer = cv2.imencode(".jpg", result["annotated_frame"], [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_b64 = base64.b64encode(buffer).decode("utf-8")

            # Build SSE event
            event_data = {
                "frame_base64": frame_b64,
                "avg_count": result["avg_count"],
                "frame_count": result["frame_count"],
                "yolo_count": result.get("yolo_count", 0),
                "cnn_count": result.get("cnn_count", 0),
                "model_used": result["model_used"],
                "frame_number": frame_number,
                "total_frames": total_frames,
                "progress": round(frame_number / max(total_frames, 1) * 100, 1),
                "threshold": result["threshold"],
            }

            yield f"data: {json.dumps(event_data)}\n\n"

        cap.release()

        # Send completion event
        yield f"data: {json.dumps({'done': True, 'total_processed': processed})}\n\n"

        # Clean up uploaded file
        try:
            os.remove(video_path)
        except OSError:
            pass

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    print("=" * 60)
    print("  Crowd Detection Server")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60)

    # Start loading models in background thread
    model_thread = threading.Thread(target=load_models_background, daemon=True)
    model_thread.start()

    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
