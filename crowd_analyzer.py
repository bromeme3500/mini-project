"""
Crowd Analyzer Module
Runs both YOLO and CNN on every frame, takes the higher count.
This eliminates the unreliable threshold-based switching problem.
"""

from collections import deque
import cv2
import numpy as np

from yolo_detector import YOLODetector
from cnn_counter import CNNCounter


class CrowdAnalyzer:
    """
    Dual-model crowd analyzer — always runs both YOLO and CNN,
    takes the higher count to ensure dense crowds are never missed.

    - YOLO: Fast individual detection, accurate for sparse crowds
    - CNN (CSRNet): Density estimation, accurate for dense crowds
    - Final count = YOLO if YOLO count <= 10 else CNN count

    Visualization:
    - If YOLO count wins → show bounding boxes (useful for sparse scenes)
    - If CNN count wins → show density heatmap (useful for dense scenes)
    """

    def __init__(self, threshold=20):
        """
        Initialize the crowd analyzer.

        Args:
            threshold: Controls visualization only — when CNN count is above
                       this, use heatmap overlay instead of bounding boxes.
                       No longer affects which model's count is used.
        """
        self.threshold = threshold
        self.yolo = YOLODetector()
        self.cnn = CNNCounter()
        self.count_history = deque(maxlen=10)  # Rolling window of last 10 frame counts

    def analyze_frame(self, frame):
        """
        Analyze a single frame using both models.

        Steps:
            1. Run YOLO → get individual person count + bounding boxes
            2. Run CNN → get density-based count + heatmap
            3. Take max(yolo, cnn) as the final count
            4. Choose visualization based on which model won
            5. Update rolling average

        Args:
            frame: BGR image (numpy array)

        Returns:
            dict with keys:
                - avg_count: float, rolling average of last 10 frames
                - frame_count: int, count for this specific frame
                - yolo_count: int, YOLO's individual count
                - cnn_count: int, CNN's density count
                - model_used: str, which model's count was higher
                - annotated_frame: frame with visual annotations
                - threshold: int, the current threshold
        """
        # Step 1: Run YOLO (fast, ~0.2s)
        yolo_count, yolo_frame, boxes = self.yolo.detect(frame)

        # Step 2: Run CNN (slower, ~2-4s on CPU)
        if self.cnn.use_multiscale:
            cnn_count, cnn_frame = self.cnn.count_multiscale(frame)
        else:
            cnn_count, cnn_frame = self.cnn.count(frame)

        # Step 3 & 4: Choose model and visualization based on count threshold
        if yolo_count <= 10:
            # Sparse crowd, trust YOLO and show bounding boxes
            frame_count = yolo_count
            annotated_frame = yolo_frame
            model_used = "YOLO"
        else:
            # Dense crowd, switch to CNN and show heatmap
            frame_count = cnn_count
            annotated_frame = cnn_frame
            model_used = "CNN"

        # Step 5: Update rolling average
        self.count_history.append(frame_count)
        avg_count = self._weighted_average()

        # Draw info overlay showing both counts
        annotated_frame = self._draw_info_overlay(
            annotated_frame, frame_count, avg_count, model_used,
            yolo_count, cnn_count
        )

        return {
            "avg_count": round(avg_count, 1),
            "frame_count": frame_count,
            "yolo_count": yolo_count,
            "cnn_count": cnn_count,
            "model_used": model_used,
            "annotated_frame": annotated_frame,
            "threshold": self.threshold,
        }

    def _weighted_average(self):
        """
        Calculate weighted rolling average giving more weight to recent frames.
        This makes the count more responsive to changes while still being smooth.
        """
        if not self.count_history:
            return 0.0
        n = len(self.count_history)
        # Linear weights: oldest=1, newest=n
        weights = list(range(1, n + 1))
        weighted_sum = sum(c * w for c, w in zip(self.count_history, weights))
        return weighted_sum / sum(weights)

    def _draw_info_overlay(self, frame, frame_count, avg_count, model_used,
                           yolo_count, cnn_count):
        """Draw a semi-transparent info panel showing both model counts."""
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Semi-transparent dark rectangle (taller to fit both counts)
        panel_h = 130
        cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # Colors
        color_yolo = (0, 255, 0)    # Green
        color_cnn = (0, 165, 255)   # Orange
        color_white = (255, 255, 255)
        color_winner = color_yolo if model_used == "YOLO" else color_cnn

        # Line 1: Winner model + final count
        cv2.putText(
            frame,
            f"Count: {frame_count}  (via {model_used})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color_winner,
            2,
        )

        # Line 2: Both model counts side by side
        cv2.putText(
            frame,
            f"YOLO: {yolo_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color_yolo,
            2,
        )
        cv2.putText(
            frame,
            f"CNN: {cnn_count}",
            (200, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color_cnn,
            2,
        )

        # Line 3: Rolling average
        cv2.putText(
            frame,
            f"Avg (10 frames): {avg_count:.1f}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color_white,
            2,
        )

        # Line 4: Method explanation
        cv2.putText(
            frame,
            "YOLO if count <= 10 else CNN",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
        )

        return frame

    def reset(self):
        """Reset the rolling average history for a new video."""
        self.count_history.clear()

    def set_threshold(self, new_threshold):
        """Update the threshold (used for visualization preference)."""
        self.threshold = max(1, int(new_threshold))
