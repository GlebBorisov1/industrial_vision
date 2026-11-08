from __future__ import annotations
import logging
import cv2
import numpy as np

from src.inference.types import Detection

logger = logging.getLogger(__name__)


class SpillDetector:
    """
    Исследовательский эвристический детектор разлива жидкости.
    НЕ является заменой segmentation. Используется для демонстрации сквозного конвейера.

    Логика:
      1. Конвертация в HSV.
      2. Маска тёмных/масляных пятен (низкая насыщенность + средняя яркость).
      3. Морфологические операции (убрать шум).
      4. Поиск контуров; фильтрация по площади и соотношению сторон.
      5. Возврат Detection с class_name="spill" (bbox вокруг контура).
    """

    def __init__(self,
                 min_area_px: int = 5000,
                 max_aspect_ratio: float = 6.0,
                 hsv_low: tuple[int, int, int] = (0, 0, 20),
                 hsv_high: tuple[int, int, int] = (180, 80, 90)):
        self.min_area_px = min_area_px
        self.max_aspect_ratio = max_aspect_ratio
        self.hsv_low = np.array(hsv_low, dtype=np.uint8)
        self.hsv_high = np.array(hsv_high, dtype=np.uint8)
        logger.info(f"SpillDetector инициализирован: min_area={min_area_px}px², "
                    f"HSV=[{hsv_low}..{hsv_high}]")

    def detect(self, frame) -> list[Detection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_low, self.hsv_high)

        # Убираем шум: открытие (удаляет мелкие пятна), закрытие (заполняет дыры)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[Detection] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area_px:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = max(w, h) / max(1, min(w, h))
            if aspect > self.max_aspect_ratio:
                continue

            # Эвристическая уверенность: чем крупнее пятно, тем выше (потолок 1.0)
            confidence = min(1.0, area / (self.min_area_px * 3))

            detections.append(Detection(
                class_id=-1,
                class_name="spill",
                confidence=confidence,
                x1=x, y1=y, x2=x + w, y2=y + h,
            ))

        return detections
