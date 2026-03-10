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
        self.model.to(self.device)
        self.conf = config.confidence_threshold
        self.iou = config.iou_threshold

    def detect_batch(self, frames: list[np.ndarray], frame_indices: list[int], court_bbox=None) -> list[list[Detection]]:
        """Run batched detection on a list of frames."""
        import cv2
        batch_inputs = []
        croppings = []
        original_sizes = []

        for frame in frames:
            original_sizes.append((frame.shape[1], frame.shape[0])) # w, h
            if court_bbox:
                y1, y2, x1, x2 = court_bbox
                cropped = frame[y1:y2, x1:x2]
                croppings.append((x1, y1))
            else:
                cropped = frame
                croppings.append((0, 0))
                
            frame_small = cv2.resize(cropped, (960, 540))
            batch_inputs.append(frame_small)

        results = self.model.predict(
            batch_inputs,
            conf=self.conf,
            iou=self.iou,
            max_det=20,
            device=self.device,
            classes=[self.PERSON_CLASS, self.BALL_CLASS],
            verbose=False,
        )

        batch_detections = []
        for i, result in enumerate(results):
            detections = []
            boxes = result.boxes
            cx, cy = croppings[i]
            orig_w, orig_h = original_sizes[i]
            
            # scaling factor from 960x540 to the cropped frame size
            if court_bbox:
                crop_w = court_bbox[3] - court_bbox[2]
                crop_h = court_bbox[1] - court_bbox[0]
            else:
                crop_w, crop_h = orig_w, orig_h
                
            scale_x = crop_w / 960.0
            scale_y = crop_h / 540.0

            for j in range(len(boxes)):
                xyxy = boxes.xyxy[j].cpu().numpy()
                conf = float(boxes.conf[j].cpu().numpy())
                cls_id = int(boxes.cls[j].cpu().numpy())
                cls_name = result.names[cls_id]

                # scale and offset
                x1 = (xyxy[0] * scale_x) + cx
                y1 = (xyxy[1] * scale_y) + cy
                x2 = (xyxy[2] * scale_x) + cx
                y2 = (xyxy[3] * scale_y) + cy

                detections.append(Detection(
                    bbox=BoundingBox(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                    ),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                    frame_idx=frame_indices[i],
                ))
            batch_detections.append(detections)

        return batch_detections
