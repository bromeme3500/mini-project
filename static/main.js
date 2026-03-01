/**
 * CrowdSense — Frontend Controller
 * Handles video upload, SSE streaming, and real-time UI updates.
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

    let eventSource = null;

    // ---- Slider Updates ----
    thresholdSlider.addEventListener("input", () => {
        thresholdValue.textContent = thresholdSlider.value;
    });
    skipSlider.addEventListener("input", () => {
        skipValue.textContent = skipSlider.value;
    });

    // ---- Drag & Drop ----
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
        if (videoInput.files.length > 0) {
            uploadVideo(videoInput.files[0]);
        }
    });

    // ---- Upload ----
    async function uploadVideo(file) {
        const formData = new FormData();
        formData.append("video", file);

        // Show processing section
        uploadSection.classList.add("hidden");
        processingSection.classList.remove("hidden");
        canvasOverlay.classList.remove("hidden");
        resetStats();

        try {
            const response = await fetch("/upload", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const err = await response.json();
                alert("Upload failed: " + (err.error || "Unknown error"));
                showUploadSection();
                return;
            }

            const data = await response.json();
            startProcessing(data.video_id, data.total_frames);
        } catch (error) {
            alert("Upload failed: " + error.message);
            showUploadSection();
        }
    }

    // ---- SSE Processing ----
    function startProcessing(videoId, totalFrames) {
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
                showUploadSection();
                return;
            }

            // Handle model loading status
            if (data.status === "loading_models") {
                const overlayText = canvasOverlay.querySelector("p");
                if (overlayText) {
                    overlayText.textContent = data.message;
                }
                return;
            }

            if (data.done) {
                onProcessingComplete(data.total_processed);
                return;
            }

            // Hide overlay on first frame
            if (firstFrame) {
                canvasOverlay.classList.add("hidden");
                firstFrame = false;
            }

            // Draw frame to canvas
            drawFrame(data.frame_base64);

            // Update stats
            updateStats(data);
        };

        eventSource.onerror = () => {
            // SSE connection closed (normal at end of stream)
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }
        };
    }

    // ---- Draw Frame ----
    function drawFrame(base64Data) {
        const img = new Image();
        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
        };
        img.src = "data:image/jpeg;base64," + base64Data;
    }

    // ---- Animated Number Transition ----
    function animateNumber(el, newValue) {
        const current = parseFloat(el.textContent) || 0;
        const target = parseFloat(newValue);
        if (current === target) return;
        el.textContent = newValue;
        el.style.transform = "scale(1.1)";
        el.style.transition = "transform 0.2s ease";
        setTimeout(() => {
            el.style.transform = "scale(1)";
        }, 200);
    }

    // ---- Update Stats ----
    function updateStats(data) {
        // Hero count
        animateNumber(avgCountEl, data.avg_count);
        avgCountEl.classList.add("active");

        // Frame count
        frameCountEl.textContent = data.frame_count;

        // YOLO vs CNN individual counts
        if (yoloCountEl) animateNumber(yoloCountEl, data.yolo_count || 0);
        if (cnnCountEl) animateNumber(cnnCountEl, data.cnn_count || 0);

        // Model badge — show which model's count won
        modelBadge.textContent = data.model_used;
        modelBadge.className = "stat-pill-value model-badge";
        if (data.model_used === "YOLO") {
            modelBadge.classList.add("yolo");
        } else {
            modelBadge.classList.add("cnn");
        }

        // Model description
        if (modelDesc) {
            modelDesc.textContent = `YOLO: ${data.yolo_count || 0} | CNN: ${data.cnn_count || 0}`;
        }

        // Progress
        progressPercent.textContent = data.progress + "%";
        progressBar.style.width = data.progress + "%";
        frameInfo.textContent = `Frame ${data.frame_number} / ${data.total_frames}`;
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
        frameInfo.textContent = "Frame 0 / 0";
    }

    // ---- Processing Complete ----
    function onProcessingComplete(totalProcessed) {
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
        progressPercent.textContent = "100%";
        progressBar.style.width = "100%";
        if (modelDesc) modelDesc.textContent = `Done — ${totalProcessed} frames analyzed`;
    }

    // ---- Controls ----
    stopBtn.addEventListener("click", stopProcessing);
    newVideoBtn.addEventListener("click", showUploadSection);

    function stopProcessing() {
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
    }

    function showUploadSection() {
        stopProcessing();
        processingSection.classList.add("hidden");
        uploadSection.classList.remove("hidden");
        videoInput.value = "";
        resetStats();
    }
})();
