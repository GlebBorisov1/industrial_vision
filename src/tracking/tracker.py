from __future__ import annotations
import logging
from ultralytics import YOLO
from src.inference.types import Detection

logger = logging.getLogger(__name__)


class Tracker:
    """
    Обёртка над Ultralytics tracking (ByteTrack по умолчанию).
    Использует persist=True, чтобы Track ID сохранялись между кадрами.
    """

    def __init__(self, model: YOLO, confidence: float = 0.5,
                 tracker_cfg: str = "bytetrack.yaml"):
        self.model = model
        self.confidence = confidence
        self.tracker_cfg = tracker_cfg
        logger.info(f"Трекер инициализирован: {tracker_cfg}")

    def track(self, frame) -> list[Detection]:
        results = self.model.track(
            frame,
            conf=self.confidence,
            persist=True,
            tracker=self.tracker_cfg,
            verbose=False,
        )[0]

        names = results.names
        detections: list[Detection] = []

        if results.boxes is None:
            return detections

        for box in results.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Track ID может отсутствовать на первом кадре объекта
            track_id = int(box.id.item()) if box.id is not None else None

            detections.append(Detection(
                class_id=cls_id,
                class_name=str(names[cls_id]),
                confidence=conf,
                x1=x1, y1=y1, x2=x2, y2=y2,
                track_id=track_id,
            ))

        return detections
