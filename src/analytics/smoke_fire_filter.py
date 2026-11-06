from __future__ import annotations
import logging
from dataclasses import dataclass

from src.inference.types import Detection

logger = logging.getLogger(__name__)


@dataclass
class FilteredDetection:
    det: Detection
    area_px: int
    aspect_ratio: float
    passed: bool
    reason: str


def filter_smoke_fire(detections: list[Detection],
                      min_area_px: int = 2000,
                      max_aspect_ratio: float = 5.0) -> list[FilteredDetection]:
    """
    Фильтрует детекции smoke/fire по геометрическим признакам.

    Hard negatives для fire: сварка, солнце, фары, лампы — часто узкие/яркие пятна.
    Hard negatives для smoke: пар, туман, пыль — часто мелкие или очень вытянутые.

    Фильтры:
      - area_px < min_area_px → отбрасываем (слишком мелкое, скорее артефакт)
      - aspect_ratio > max_aspect_ratio → отбрасываем (слишком вытянутое, скорее блик)
    """
    results = []
    for det in detections:
        w = det.x2 - det.x1
        h = det.y2 - det.y1
        area = w * h
        aspect = max(w, h) / max(1, min(w, h))

        passed = True
        reason = "ok"

        if area < min_area_px:
            passed = False
            reason = f"too_small (area={area}px² < {min_area_px})"
        elif aspect > max_aspect_ratio:
            passed = False
            reason = f"too_stretched (aspect={aspect:.1f} > {max_aspect_ratio})"

        if not passed:
            logger.debug(f"Smoke/fire отброшен: {det.class_name} conf={det.confidence:.2f} — {reason}")

        results.append(FilteredDetection(
            det=det, area_px=area, aspect_ratio=aspect,
            passed=passed, reason=reason,
        ))
    return results
