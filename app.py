"""
Flask Application — Crowd Detection Server
Handles video upload and live DroidCam streaming via Server-Sent Events.
Supports both USB (virtual webcam index) and Wi-Fi MJPEG (DroidCam IP) modes.
"""

import os
import uuid
import json
import base64
import time
import threading
import urllib.request
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, send_file

from crowd_analyzer import CrowdAnalyzer

# Telegram Alert Configuration
TELEGRAM_BOT_TOKEN = ""  # Setup complete
TELEGRAM_CHAT_ID = "1446246146"       # Fixed: Removed the negative sign
LAST_ALERT_TIME = 0.0
ALERT_COOLDOWN = 30.0  # Seconds between alerts

def send_telegram_alert(count):
    """Sends a Telegram alert if the cooldown period has passed."""
    global LAST_ALERT_TIME
    current_time = time.time()
    
    # Ensure alerts are sent at most once every 30 seconds
    if current_time - LAST_ALERT_TIME < ALERT_COOLDOWN:
        return
        
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        print("[Telegram Alert] Token not configured. Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
        return
        
    LAST_ALERT_TIME = current_time
    
    def _send():
        try:
            message = (
                "⚠ Crowd Alert!\n"
                "Crowd limit exceeded.\n"
                f"People detected more than the threshold: {int(count)}\n"
                "Location: Camera 1"
            )
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            data = json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": message}).encode("utf-8")
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    print(f"[Telegram Alert] Sent successfully! (Count: {int(count)})")
        except Exception as e:
            print(f"[Telegram Alert] Error sending message: {e}")
            
    # Send in a background thread to prevent blocking video stream
    threading.Thread(target=_send, daemon=True).start()

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

# Live stream state
live_active = False
live_lock = threading.Lock()
live_params = {
    "device_index": 0,
    "droidcam_ip": "",
    "threshold": 20,
    "alert_limit": 10,
}


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


@app.route("/video/<video_id>")
def serve_video(video_id):
    """Serve the raw uploaded video file for browser playback."""
    video_path = None
    for f in os.listdir(app.config["UPLOAD_FOLDER"]):
        if f.startswith(video_id):
            video_path = os.path.join(app.config["UPLOAD_FOLDER"], f)
            break
    if video_path is None:
        return jsonify({"error": "Video not found"}), 404
    return send_file(video_path, conditional=True)


@app.route("/process/<video_id>")
def process_video(video_id):
    """
    Stream processed frames as Server-Sent Events.
    Each event contains: base64-encoded annotated frame, counts, model info.
    """
    threshold = request.args.get("threshold", 20, type=int)
    alert_limit = request.args.get("alert_limit", 10, type=int)
    interval = request.args.get("interval", 1.0, type=float)

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

        # Target configured interval based on video metadata
        skip_frames = max(1, int(round(fps * interval)))
        print(f"[Server] Video FPS: {fps}. Interval: {interval}s. Skip factor: {skip_frames}.")

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

            # Resize for FASTER streaming (720px width is perfect for web)
            h, w = frame.shape[:2]
            max_dim = 720
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            # Analyze the frame
            result = crowd_analyzer.analyze_frame(frame)
            processed += 1

            # Encode annotated frame as JPEG base64
            _, buffer = cv2.imencode(".jpg", result["annotated_frame"], [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_b64 = base64.b64encode(buffer).decode("utf-8")

            is_alert = result["avg_count"] >= alert_limit
            if is_alert:
                send_telegram_alert(result["avg_count"])

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
                "alert": is_alert,
                "alert_limit": alert_limit,
            }

            yield f"data: {json.dumps(event_data)}\n\n"

        cap.release()

        # Send completion event (file kept on disk so browser video player keeps working)
        yield f"data: {json.dumps({'done': True, 'total_processed': processed})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

# Live camera background capture
camera_thread = None
latest_frame = None
latest_frame_time = 0
camera_capture_lock = threading.Lock()
camera_active = False


def camera_capture_loop():
    """Continuously captures frames from the camera into a fast shared buffer."""
    global latest_frame, latest_frame_time, camera_active

    with live_lock:
        params = dict(live_params)

    if params["droidcam_ip"]:
        cam_source = f"http://{params['droidcam_ip']}:4747/video"
    else:
        cam_source = params["device_index"]

    if isinstance(cam_source, int):
        cap = cv2.VideoCapture(cam_source + cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(cam_source)

    if not cap.isOpened():
        print(f"[Camera Error] Cannot open source: {cam_source}")
        camera_active = False
        return

    print(f"[Camera Started] Source: {cam_source}")

    try:
        while camera_active:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Resize the raw frame slightly to guarantee performance for web MJPEG streaming
            h, w = frame.shape[:2]
            max_dim = 800
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            with camera_capture_lock:
                latest_frame = frame.copy()
                latest_frame_time = time.time()

            # Small sleep to limit capture to ~30-40 fps max and reduce CPU load
            time.sleep(0.02)
    finally:
        cap.release()
        print("[Camera Stopped]")


@app.route("/start_live", methods=["POST"])
def start_live():
    """
    Start a live DroidCam session.
    Accepts JSON: device_index, droidcam_ip, threshold, alert_limit, interval.
    """
    global live_active, live_params, camera_active, camera_thread
    data = request.get_json(force=True, silent=True) or {}
    
    with live_lock:
        live_params["device_index"] = int(data.get("device_index", 1))
        live_params["droidcam_ip"] = data.get("droidcam_ip", "").strip()
        live_params["threshold"] = int(data.get("threshold", 20))
        live_params["alert_limit"] = int(data.get("alert_limit", 10))
        live_params["interval"] = float(data.get("interval", 1.0))
        
        if not camera_active:
            camera_active = True
            live_active = True
            camera_thread = threading.Thread(target=camera_capture_loop, daemon=True)
            camera_thread.start()
            
    return jsonify({"status": "started"})


@app.route("/stop_live", methods=["POST"])
def stop_live():
    """Stop the live DroidCam stream."""
    global live_active, camera_active
    with live_lock:
        live_active = False
        camera_active = False
    return jsonify({"status": "stopped"})


@app.route("/live_raw_stream")
def live_raw_stream():
    """
    MJPEG fast stream of the raw camera frames.
    Provides a completely smooth native camera experience.
    """
    def generate_mjpeg():
        last_yield_time = 0
        while camera_active:
            with camera_capture_lock:
                frame = latest_frame
                ftime = latest_frame_time
                
            # Only encode and yield if we have a new frame
            if frame is not None and ftime > last_yield_time:
                last_yield_time = ftime
                ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                time.sleep(0.01)

    return Response(
        generate_mjpeg(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route("/live_stream")
def live_stream():
    """
    SSE stream for backend detection data only.
    Reads from the shared background camera buffer at the configured interval.
    """
    alert_limit = request.args.get("alert_limit", 10, type=int)
    interval = request.args.get("interval", 1.0, type=float)

    def generate():
        global live_active

        # Wait for models
        if not models_ready:
            yield f"data: {json.dumps({'status': 'loading_models', 'message': 'Loading AI models...'})}\n\n"
            timeout = 120
            start = time.time()
            while not models_ready and (time.time() - start) < timeout:
                time.sleep(1)
                yield f"data: {json.dumps({'status': 'loading_models', 'message': f'Loading models... ({int(time.time()-start)}s)'})}\n\n"
            if not models_ready:
                yield f"data: {json.dumps({'error': 'Model loading timed out.'})}\n\n"
                return

        # Wait for camera thread to capture the first frame
        timeout = 15
        start_wait = time.time()
        while camera_active and latest_frame is None and (time.time() - start_wait) < timeout:
            time.sleep(0.5)
            
        if not camera_active or latest_frame is None:
            yield f"data: {json.dumps({'error': 'Cannot access camera feed. Make sure DroidCam is connected.'})}\n\n"
            return

        crowd_analyzer = get_analyzer(live_params.get("threshold", 20))
        crowd_analyzer.reset()
        frame_number = 0
        last_proc_time = 0

        try:
            while camera_active and live_active:
                with live_lock:
                    current_alert_limit = int(live_params.get("alert_limit", 10))
                    
                # Check for configured interval
                current_time = time.time()
                if current_time - last_proc_time < interval:
                    time.sleep(0.1)
                    continue
                
                with camera_capture_lock:
                    frame = latest_frame
                    ftime = latest_frame_time
                    
                if frame is None or ftime <= last_proc_time:
                    time.sleep(0.1)
                    continue
                    
                last_proc_time = current_time
                frame_number += 1

                # Resize for detection (analyzer expects standard size)
                h, w = frame.shape[:2]
                max_dim = 720
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

                result = crowd_analyzer.analyze_frame(frame)
                alert_triggered = result["avg_count"] >= current_alert_limit
                
                if alert_triggered:
                    send_telegram_alert(result["avg_count"])

                event_data = {
                    "avg_count": result["avg_count"],
                    "frame_count": result["frame_count"],
                    "yolo_count": result.get("yolo_count", 0),
                    "cnn_count": result.get("cnn_count", 0),
                    "model_used": result["model_used"],
                    "frame_number": frame_number,
                    "threshold": result["threshold"],
                    "alert": alert_triggered,
                    "alert_limit": current_alert_limit,
                }
                yield f"data: {json.dumps(event_data)}\n\n"
        finally:
            yield f"data: {json.dumps({'done': True})}\n\n"

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
    print("  Crowd Detection Server — DroidCam Edition")
    print("  Open http://localhost:5000 in your browser")
    print("  Connect DroidCam via USB or Wi-Fi")
    print("=" * 60)

    # Start loading models in background thread
    model_thread = threading.Thread(target=load_models_background, daemon=True)
    model_thread.start()

    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True, use_reloader=False)
