"""
YOLOv8 Person Detector Module
Uses ultralytics YOLOv8n for detecting individual people in frames.
"""

from ultralytics import YOLO
import cv2
import numpy as np


class YOLODetector:
    """Wrapper around YOLOv8 for person detection."""

    def __init__(self, model_path="yolov8n.pt", confidence=0.3):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path: Path to YOLOv8 model weights (auto-downloads if not present)
            confidence: Minimum confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.PERSON_CLASS_ID = 0  # COCO class ID for 'person'

    def detect(self, frame):
        """
        Detect people in a frame.

        Args:
            frame: BGR image (numpy array from OpenCV)

        Returns:
            tuple: (person_count, annotated_frame, boxes)
                - person_count: int, number of people detected
                - annotated_frame: frame with bounding boxes drawn
                - boxes: list of [x1, y1, x2, y2, confidence] for each person
        """
        results = self.model(frame, conf=self.confidence, verbose=False)
        
        annotated_frame = frame.copy()
        person_boxes = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id == self.PERSON_CLASS_ID:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    person_boxes.append([x1, y1, x2, y2, conf])

                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Person {conf:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0], y1),
                        (0, 255, 0),
                        -1,
                    )
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                    )

        return len(person_boxes), annotated_frame, person_boxes
