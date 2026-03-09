"""Shared detection data types."""

from dataclasses import dataclass


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.width, self.height)

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class Detection:
    bbox: BoundingBox
    confidence: float
    class_id: int  # 0=person, 32=sports ball (COCO)
    class_name: str
    frame_idx: int
