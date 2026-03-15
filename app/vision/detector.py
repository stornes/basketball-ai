"""Player and ball detection using YOLOv8."""

import numpy as np

from app.pipeline.pipeline_config import PipelineConfig
from app.vision.detection_types import BoundingBox, Detection


class PlayerBallDetector:
    """Wraps YOLOv8 for detecting persons and sports balls.

    Supports both COCO pre-trained models (person=0, ball=32) and
    fine-tuned models with custom class maps via config.class_map.
    """

    # COCO defaults
    PERSON_CLASS = 0
    BALL_CLASS = 32  # COCO "sports ball"

    def __init__(self, config: PipelineConfig):
        from ultralytics import YOLO
        self.model = YOLO(config.yolo_model)
        self.device = config.device
        self.model.to(self.device)
        self.conf = config.confidence_threshold
        self.iou = config.iou_threshold
        self._setup_class_map(config.class_map)

    def _setup_class_map(self, class_map: dict | None):
        """Configure class ID mapping for detection filtering.

        For COCO models: person=0, ball=32.
        For fine-tuned models: class_map maps model IDs to roles,
        e.g. {0: "ball", 1: "player", 2: "hoop"}.
        """
        if class_map:
            self._class_filter = list(class_map.keys())
            self._person_ids = {
                k for k, v in class_map.items() if v.lower() in ("person", "player")
            }
            self._ball_ids = {
                k for k, v in class_map.items() if v.lower() in ("ball", "basketball")
            }
        else:
            self._class_filter = [self.PERSON_CLASS, self.BALL_CLASS]
            self._person_ids = {self.PERSON_CLASS}
            self._ball_ids = {self.BALL_CLASS}

    def _normalize_class_id(self, raw_id: int) -> int:
        """Map model class ID to canonical COCO IDs (person=0, ball=32).

        This ensures downstream code (tracker, shot detector) works
        unchanged regardless of whether the model uses COCO or custom IDs.
        """
        if raw_id in self._person_ids:
            return self.PERSON_CLASS
        if raw_id in self._ball_ids:
            return self.BALL_CLASS
        return raw_id

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
            classes=self._class_filter,
            verbose=False,
        )

        # Pre-compute court crop dimensions (invariant across frames)
        if court_bbox:
            court_crop_w = court_bbox[3] - court_bbox[2]
            court_crop_h = court_bbox[1] - court_bbox[0]

        batch_detections = []
        for i, result in enumerate(results):
            detections = []
            boxes = result.boxes
            cx, cy = croppings[i]
            orig_w, orig_h = original_sizes[i]

            crop_w = court_crop_w if court_bbox else orig_w
            crop_h = court_crop_h if court_bbox else orig_h
            scale_x = crop_w / 960.0
            scale_y = crop_h / 540.0

            # Transfer all box data from GPU once per frame (not per box)
            if len(boxes) == 0:
                batch_detections.append([])
                continue
            all_xyxy = boxes.xyxy.cpu().numpy()
            all_conf = boxes.conf.cpu().numpy()
            all_cls = boxes.cls.cpu().numpy().astype(int)
            names = result.names

            for j in range(len(all_cls)):
                raw_cls_id = int(all_cls[j])
                cls_id = self._normalize_class_id(raw_cls_id)

                # scale and offset
                x1 = (all_xyxy[j, 0] * scale_x) + cx
                y1 = (all_xyxy[j, 1] * scale_y) + cy
                x2 = (all_xyxy[j, 2] * scale_x) + cx
                y2 = (all_xyxy[j, 3] * scale_y) + cy

                detections.append(Detection(
                    bbox=BoundingBox(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                    ),
                    confidence=float(all_conf[j]),
                    class_id=cls_id,
                    class_name=names[raw_cls_id],
                    frame_idx=frame_indices[i],
                ))
            batch_detections.append(detections)

        return batch_detections
