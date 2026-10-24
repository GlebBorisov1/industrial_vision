from __future__ import annotations
import logging
from ultralytics import YOLO
from .types import Detection

logger = logging.getLogger(__name__)

class Detector:
    def __init__(self, model_path: str, confidence: float = 0.5):
        self.confidence = confidence
        logger.info(f"Загрузка модели: {model_path}")

        try:
            self.model = YOLO(model_path)
            logger.info("Модель успешно загружена.")
        except Exception as e:
            logger.critical(f"Ошибка загрузки модели: {e}")
            raise

    def predict(self, frame) -> list[Detection]:
        result = self.model.predict(frame, conf=self.confidence, verbose=False)[0]
        names = result.names
        detections: list[Detection] = []

        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            detections.append(Detection(
                class_id=cls_id,
                class_name=str(names[cls_id]),
                confidence=conf,
                x1=x1, y1=y1, x2=x2, y2=y2
            ))

        return detections
