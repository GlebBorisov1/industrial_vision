from __future__ import annotations
import logging
from dataclasses import dataclass

from src.inference.types import Detection

logger = logging.getLogger(__name__)


@dataclass
class PPECheckResult:
    person_track_id: int | None
    person_bbox: tuple[int, int, int, int]
    has_helmet: bool | None
    has_vest: bool | None
    helmet_confidence: float
    vest_confidence: float
    eligible: bool               # человек достаточно крупный для анализа
    reason: str                  # для логов: "ok", "too_small", "unknown_helmet"


def _bbox_contains(outer: tuple[int, int, int, int],
                   inner: tuple[int, int, int, int],
                   padding: float = 0.0) -> bool:
    """
    Проверяет, находится ли внутренний bbox внутри внешнего (с запасом padding по долям).
    Используется для сопоставления helmet/head/vest с bbox person.
    """
    ox1, oy1, ox2, oy2 = outer
    w, h = ox2 - ox1, oy2 - oy1
    pad_x, pad_y = int(w * padding), int(h * padding)

    return (inner[0] >= ox1 - pad_x and inner[1] >= oy1 - pad_y
            and inner[2] <= ox2 + pad_x and inner[3] <= oy2 + pad_y)


def _center_inside(outer: tuple[int, int, int, int],
                   inner: tuple[int, int, int, int]) -> bool:
    """Центр внутреннего bbox попадает во внешний."""
    cx = (inner[0] + inner[2]) // 2
    cy = (inner[1] + inner[3]) // 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def check_ppe_for_person(person: Detection,
                         ppe_detections: list[Detection],
                         *,
                         min_person_height: int = 120,
                         bbox_padding: float = 0.10) -> PPECheckResult:
    """
    Проверяет наличие каски и жилета у одного человека.

    Логика каски:
      - есть helmet в bbox person  → has_helmet = True
      - есть head в bbox person    → has_helmet = False (более надёжный сигнал нарушения)
      - ни helmet, ни head         → has_helmet = None (неизвестно, событие не генерируем)

    Логика жилета:
      - есть vest в bbox person    → has_vest = True
      - нет vest                   → has_vest = False

    Слишком маленький человек (height < min_person_height) → eligible=False.
    """
    person_bbox = (person.x1, person.y1, person.x2, person.y2)
    height = person.y2 - person.y1

    if height < min_person_height:
        return PPECheckResult(
            person_track_id=person.track_id,
            person_bbox=person_bbox,
            has_helmet=None, has_vest=None,
            helmet_confidence=0.0, vest_confidence=0.0,
            eligible=False, reason=f"too_small (h={height}px < {min_person_height}px)",
        )

    # Отбираем PPE-детекции, пространственно связанные с этим человеком
    helmet_dets = [d for d in ppe_detections
                   if d.class_name == "helmet"
                   and (_center_inside(person_bbox, (d.x1, d.y1, d.x2, d.y2))
                        or _bbox_contains(person_bbox, (d.x1, d.y1, d.x2, d.y2), bbox_padding))]
    head_dets = [d for d in ppe_detections
                 if d.class_name == "head"
                 and (_center_inside(person_bbox, (d.x1, d.y1, d.x2, d.y2))
                      or _bbox_contains(person_bbox, (d.x1, d.y1, d.x2, d.y2), bbox_padding))]
    vest_dets = [d for d in ppe_detections
                 if d.class_name == "vest"
                 and (_center_inside(person_bbox, (d.x1, d.y1, d.x2, d.y2))
                      or _bbox_contains(person_bbox, (d.x1, d.y1, d.x2, d.y2), bbox_padding))]

    # Логика каски
    if helmet_dets:
        has_helmet = True
        helmet_conf = max(d.confidence for d in helmet_dets)
        reason = "ok"
    elif head_dets:
        has_helmet = False
        helmet_conf = max(d.confidence for d in head_dets)
        reason = "helmet_missing (head detected)"
    else:
        has_helmet = None
        helmet_conf = 0.0
        reason = "helmet_unknown (no helmet/head in bbox)"

    # Логика жилета
    if vest_dets:
        has_vest = True
        vest_conf = max(d.confidence for d in vest_dets)
    else:
        has_vest = False
        vest_conf = 0.0

    return PPECheckResult(
        person_track_id=person.track_id,
        person_bbox=person_bbox,
        has_helmet=has_helmet,
        has_vest=has_vest,
        helmet_confidence=helmet_conf,
        vest_confidence=vest_conf,
        eligible=True,
        reason=reason,
    )
