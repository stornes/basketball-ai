"""Player and ball detection using YOLOv8."""

import numpy as np

from app.pipeline.pipeline_config import PipelineConfig
from app.vision.detection_types import BoundingBox, Detection


class PlayerBallDetector:
    """Wraps YOLOv8 for detecting persons and sports balls."""

    PERSON_CLASS = 0
    BALL_CLASS = 32  # COCO "sports ball"

    def __init__(self, config: PipelineConfig):
        from ultralytics import YOLO
        self.model = YOLO(config.yolo_model)
        self.device = config.device
        self.conf = config.confidence_threshold
        self.iou = config.iou_threshold

    def detect_frame(self, frame: np.ndarray, frame_idx: int) -> list[Detection]:
        """Run detection on a single frame."""
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            classes=[self.PERSON_CLASS, self.BALL_CLASS],
            verbose=False,
        )

        detections = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = result.names[cls_id]

                detections.append(Detection(
                    bbox=BoundingBox(
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                    ),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                    frame_idx=frame_idx,
                ))

        return detections
