/**
 * CrowdSense — Frontend Controller
 * Handles video upload, DroidCam live streaming, SSE frame rendering,
 * Web Audio buzzer alert, and real-time UI updates.
 */

(function () {
    "use strict";

    // ---- DOM Elements ----
    const dropZone = document.getElementById("dropZone");
    const videoInput = document.getElementById("videoInput");
    const thresholdSlider = document.getElementById("thresholdSlider");
    const thresholdValue = document.getElementById("thresholdValue");
    const skipSlider = document.getElementById("skipSlider");
    const skipValue = document.getElementById("skipValue");

    const uploadSection = document.getElementById("upload-section");
    const liveSection = document.getElementById("live-section");
    const processingSection = document.getElementById("processing-section");

    const avgCountEl = document.getElementById("avgCount");
    const frameCountEl = document.getElementById("frameCount");
    const yoloCountEl = document.getElementById("yoloCount");
    const cnnCountEl = document.getElementById("cnnCount");
    const modelBadge = document.getElementById("modelBadge");
    const modelDesc = document.getElementById("modelDesc");
    const progressPercent = document.getElementById("progressPercent");
    const frameInfo = document.getElementById("frameInfo");
    const progressBar = document.getElementById("progressBar");

    const canvas = document.getElementById("videoCanvas");
    const ctx = canvas.getContext("2d");
    const canvasOverlay = document.getElementById("canvasOverlay");

    const stopBtn = document.getElementById("stopBtn");
    const newVideoBtn = document.getElementById("newVideoBtn");

    // Live camera elements
    const alertLimitSlider = document.getElementById("alertLimitSlider");
    const alertLimitVal = document.getElementById("alertLimitVal");
    const liveThresholdSlider = document.getElementById("liveThresholdSlider");
    const liveThresholdVal = document.getElementById("liveThresholdVal");
    const startLiveBtn = document.getElementById("startLiveBtn");
    const stopLiveBtn = document.getElementById("stopLiveBtn");
    const liveDotIndicator = document.getElementById("liveDotIndicator");
    const liveError = document.getElementById("liveError");
    const sourceLabel = document.getElementById("sourceLabel");

    // Alert banner
    const alertBanner = document.getElementById("alertBanner");
    const alertCountDisplay = document.getElementById("alertCountDisplay");
    const silenceBtn = document.getElementById("silenceBtn");

    let eventSource = null;
    let isLiveMode = false;
    let alertSilenced = false;
    let alertActive = false;
    let buzzerIntervalId = null;
    let audioCtx = null;

    // ============================================================
    // TAB SWITCHING
    // ============================================================
    window.switchTab = function (tab) {
        document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
        if (tab === "upload") {
            document.getElementById("tabUpload").classList.add("active");
            uploadSection.classList.remove("hidden");
            liveSection.classList.add("hidden");
        } else {
            document.getElementById("tabLive").classList.add("active");
            liveSection.classList.remove("hidden");
            uploadSection.classList.add("hidden");
        }
    };

    // ============================================================
    // SLIDER UPDATES
    // ============================================================
    const liveSkipSlider = document.getElementById('liveSkipSlider');
    const liveSkipVal = document.getElementById('liveSkipVal');
    const uploadAlertLimitSlider = document.getElementById('uploadAlertLimitSlider');
    const uploadAlertLimitVal = document.getElementById('uploadAlertLimitVal');

    // Live Config Sliders
    liveThresholdSlider.addEventListener('input', (e) => {
        liveThresholdVal.textContent = e.target.value;
    });
    alertLimitSlider.addEventListener('input', (e) => {
        alertLimitVal.textContent = e.target.value;
    });
    liveSkipSlider.addEventListener('input', (e) => {
        liveSkipVal.textContent = e.target.value;
    });

    // Upload Config Sliders
    thresholdSlider.addEventListener("input", () => {
        thresholdValue.textContent = thresholdSlider.value;
    });
    skipSlider.addEventListener('input', (e) => {
        skipValue.textContent = e.target.value;
    });
    uploadAlertLimitSlider.addEventListener('input', (e) => {
        uploadAlertLimitVal.textContent = e.target.value;
    });
    // ============================================================
    // DRAG & DROP (upload tab)
    // ============================================================
    dropZone.addEventListener("click", () => videoInput.click());
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });
    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("drag-over");
    });
    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type.startsWith("video/")) {
            uploadVideo(files[0]);
        }
    });
    videoInput.addEventListener("change", () => {
        if (videoInput.files.length > 0) uploadVideo(videoInput.files[0]);
    });

    // ============================================================
    // WEB AUDIO BUZZER
    // ============================================================
    function getAudioContext() {
        if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        // Resume if suspended (browser policy)
        if (audioCtx.state === "suspended") {
            audioCtx.resume();
        }
        return audioCtx;
    }

    /**
     * Plays a short 3-beep siren burst using the Web Audio API.
     * No external file needed — synthesized on the fly.
     */
    function playBuzzer() {
        if (alertSilenced) return;
        const ctx2 = getAudioContext();

        // Two-tone siren: alternating 880 Hz and 660 Hz
        const tones = [880, 660, 880];
        tones.forEach((freq, i) => {
            const osc = ctx2.createOscillator();
            const gain = ctx2.createGain();

            osc.type = "sawtooth";
            osc.frequency.setValueAtTime(freq, ctx2.currentTime);
            gain.gain.setValueAtTime(0, ctx2.currentTime + i * 0.18);
            gain.gain.linearRampToValueAtTime(0.35, ctx2.currentTime + i * 0.18 + 0.02);
            gain.gain.exponentialRampToValueAtTime(0.001, ctx2.currentTime + i * 0.18 + 0.16);

            osc.connect(gain);
            gain.connect(ctx2.destination);

            osc.start(ctx2.currentTime + i * 0.18);
            osc.stop(ctx2.currentTime + i * 0.18 + 0.18);
        });
    }

    // ============================================================
    // ALERT BANNER
    // ============================================================
    function showAlert(count, limit) {
        if (!alertActive) {
            alertBanner.classList.remove("hidden");
            alertActive = true;
            // Repeat buzzer every 3s while alert is active
            if (!alertSilenced) {
                playBuzzer();
                buzzerIntervalId = setInterval(() => {
                    if (!alertSilenced) playBuzzer();
                }, 3000);
            }
        }
        alertCountDisplay.textContent = `${count} / ${limit} people`;
    }

    function hideAlert() {
        if (alertActive) {
            alertBanner.classList.add("hidden");
            alertActive = false;
            clearInterval(buzzerIntervalId);
            buzzerIntervalId = null;
        }
    }

    silenceBtn.addEventListener("click", () => {
        alertSilenced = true;
        clearInterval(buzzerIntervalId);
        buzzerIntervalId = null;
        silenceBtn.textContent = "✅";
        silenceBtn.title = "Alert silenced";
    });

    // ============================================================
    // UPLOAD FLOW
    // ============================================================
    async function uploadVideo(file) {
        const formData = new FormData();
        formData.append("video", file);
        formData.append("threshold", thresholdSlider.value);
        formData.append("frame_skip", skipSlider.value);
        formData.append("alert_limit", uploadAlertLimitSlider.value);

        isLiveMode = false;
        showProcessingSection("📁 Uploaded: " + file.name);

        try {
            const response = await fetch("/upload", {
                method: "POST",
                body: formData,
            });
            if (!response.ok) {
                const err = await response.json();
                alert("Upload failed: " + (err.error || "Unknown error"));
                showInputSection();
                return;
            }
            const data = await response.json();
            startVideoProcessing(data.video_id, data.total_frames);
        } catch (error) {
            alert("Upload failed: " + error.message);
            showInputSection();
        }
    }

    function startVideoProcessing(videoId, totalFrames) {
        const threshold = thresholdSlider.value;
        const skip = skipSlider.value;
        const url = `/process/${videoId}?threshold=${threshold}&skip=${skip}`;

        eventSource = new EventSource(url);
        let firstFrame = true;

        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.error) {
                alert("Processing error: " + data.error);
                stopProcessing();
                showInputSection();
                return;
            }
            if (data.status === "loading_models") {
                const overlayText = canvasOverlay.querySelector("p");
                if (overlayText) overlayText.textContent = data.message;
                return;
            }
            if (data.done) {
                onVideoComplete(data.total_processed);
                return;
            }
            if (firstFrame) {
                canvasOverlay.classList.add("hidden");
                firstFrame = false;
            }
            drawFrame(data.frame_base64);
            updateStats(data);
        };

        eventSource.onerror = () => {
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }
        };
    }

    // ============================================================
    // LIVE STREAM FLOW
    // ============================================================
    window.startLive = async function () {
        const deviceIndex = parseInt(document.getElementById("deviceIndex").value) || 0;
        const droidcamIp = document.getElementById("droidcamIp").value.trim();
        const alertLimit = parseInt(alertLimitSlider.value);
        const vizThreshold = parseInt(liveThresholdSlider.value);

        // Resume audio context on user action (browser policy requires gesture)
        getAudioContext();

        // Reset silence flag for new session
        alertSilenced = false;
        silenceBtn.textContent = "🔇";
        silenceBtn.title = "Silence alert";

        // Tell backend to start
        const liveThreshold = document.getElementById("liveThresholdSlider").value;
        const liveAlertLimit = document.getElementById("alertLimitSlider").value;
        const liveSkip = document.getElementById("liveSkipSlider").value;

        try {
            const res = await fetch("/start_live", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    device_index: deviceIndex,
                    droidcam_ip: droidcamIp,
                    threshold: liveThreshold,
                    alert_limit: liveAlertLimit,
                    frame_skip: liveSkip
                }),
            });
            if (!res.ok) throw new Error("Server rejected /start_live");
        } catch (e) {
            showLiveError("Could not reach server: " + e.message);
            return;
        }

        const sourceStr = droidcamIp
            ? `📱 DroidCam Wi-Fi — ${droidcamIp}:4747`
            : `📱 DroidCam USB — device ${deviceIndex}`;

        isLiveMode = true;
        showProcessingSection(sourceStr);

        // Update live indicator
        liveDotIndicator && liveDotIndicator.classList.add("live");
        startLiveBtn && startLiveBtn.classList.add("hidden");
        stopLiveBtn && stopLiveBtn.classList.remove("hidden");
        hideLiveError();

        // Connect to SSE
        const sseUrl = `/live_stream?alert_limit=${alertLimit}`;
        eventSource = new EventSource(sseUrl);
        let firstFrame = true;

        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);

            if (data.error) {
                showLiveError(data.error);
                stopLive();
                showInputSection();
                return;
            }
            if (data.status === "loading_models") {
                const overlayText = canvasOverlay.querySelector("p");
                if (overlayText) overlayText.textContent = data.message;
                return;
            }
            if (data.done) {
                stopLive();
                return;
            }
            if (firstFrame) {
                canvasOverlay.classList.add("hidden");
                firstFrame = false;
            }

            drawFrame(data.frame_base64);
            updateStats(data, true /* isLive */);

            // Crowd limit alert
            if (data.alert) {
                showAlert(data.avg_count, data.alert_limit);
            } else {
                hideAlert();
            }
        };

        eventSource.onerror = () => {
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }
        };
    };

    window.stopLive = async function () {
        try {
            await fetch("/stop_live", { method: "POST" });
        } catch (_) { /* ignore */ }

        stopProcessing();
        hideAlert();

        // Reset live UI
        liveDotIndicator && liveDotIndicator.classList.remove("live");
        startLiveBtn && startLiveBtn.classList.remove("hidden");
        stopLiveBtn && stopLiveBtn.classList.add("hidden");

        showInputSection();
    };

    // ============================================================
    // SHARED RENDERING
    // ============================================================
    function drawFrame(base64Data) {
        const img = new Image();
        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
        };
        img.src = "data:image/jpeg;base64," + base64Data;
    }

    function animateNumber(el, newValue) {
        if (!el) return;
        if (parseFloat(el.textContent) === parseFloat(newValue)) return;
        el.textContent = newValue;
        el.style.transform = "scale(1.1)";
        el.style.transition = "transform 0.2s ease";
        setTimeout(() => { el.style.transform = "scale(1)"; }, 200);
    }

    function updateStats(data, isLive) {
        animateNumber(avgCountEl, data.avg_count);
        avgCountEl.classList.add("active");
        frameCountEl.textContent = data.frame_count;
        if (yoloCountEl) animateNumber(yoloCountEl, data.yolo_count || 0);
        if (cnnCountEl) animateNumber(cnnCountEl, data.cnn_count || 0);

        modelBadge.textContent = data.model_used;
        modelBadge.className = "stat-pill-value model-badge";
        modelBadge.classList.add(data.model_used === "YOLO" ? "yolo" : "cnn");

        if (modelDesc) modelDesc.textContent = `YOLO: ${data.yolo_count || 0} | CNN: ${data.cnn_count || 0}`;

        if (!isLive) {
            progressPercent.textContent = (data.progress || 0) + "%";
            progressBar.style.width = (data.progress || 0) + "%";
            frameInfo.textContent = `Frame ${data.frame_number} / ${data.total_frames}`;
        } else {
            progressPercent.textContent = "LIVE";
            progressBar.style.width = "100%";
            progressBar.style.background = "linear-gradient(90deg, #ef4444, #f97316)";
            frameInfo.textContent = `Frame ${data.frame_number}`;
        }
    }

    function resetStats() {
        avgCountEl.textContent = "0";
        avgCountEl.classList.remove("active");
        frameCountEl.textContent = "0";
        if (yoloCountEl) yoloCountEl.textContent = "0";
        if (cnnCountEl) cnnCountEl.textContent = "0";
        modelBadge.textContent = "—";
        modelBadge.className = "stat-pill-value model-badge";
        if (modelDesc) modelDesc.textContent = "Running YOLO + CNN on each frame";
        progressPercent.textContent = "0%";
        progressBar.style.width = "0%";
        progressBar.style.background = "";
        frameInfo.textContent = "Frame 0 / 0";
    }

    function onVideoComplete(totalProcessed) {
        if (eventSource) { eventSource.close(); eventSource = null; }
        progressPercent.textContent = "100%";
        progressBar.style.width = "100%";
        if (modelDesc) modelDesc.textContent = `Done — ${totalProcessed} frames analyzed`;
    }

    // ============================================================
    // SECTION VISIBILITY
    // ============================================================
    function showProcessingSection(label) {
        uploadSection.classList.add("hidden");
        liveSection.classList.add("hidden");
        processingSection.classList.remove("hidden");
        canvasOverlay.classList.remove("hidden");
        if (sourceLabel) sourceLabel.textContent = label;
        resetStats();
    }

    function showInputSection() {
        stopProcessing();
        hideAlert();
        processingSection.classList.add("hidden");
        if (isLiveMode) {
            liveSection.classList.remove("hidden");
            uploadSection.classList.add("hidden");
            switchTab("live");
        } else {
            uploadSection.classList.remove("hidden");
            liveSection.classList.add("hidden");
            switchTab("upload");
        }
        videoInput.value = "";
        resetStats();
    }

    // ============================================================
    // CONTROLS
    // ============================================================
    stopBtn.addEventListener("click", () => {
        if (isLiveMode) {
            stopLive();
        } else {
            stopProcessing();
            showInputSection();
        }
    });

    newVideoBtn.addEventListener("click", () => {
        if (isLiveMode) {
            stopLive();
        } else {
            showInputSection();
        }
    });

    function stopProcessing() {
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
    }

    function showLiveError(msg) {
        if (liveError) {
            liveError.textContent = "⚠️ " + msg;
            liveError.classList.remove("hidden");
        }
    }

    function hideLiveError() {
        if (liveError) liveError.classList.add("hidden");
    }
})();
