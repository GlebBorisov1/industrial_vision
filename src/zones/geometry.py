from __future__ import annotations
import cv2
import numpy as np
from src.inference.types import Detection


def bottom_center(det: Detection) -> tuple[int, int]:
    """Нижняя центральная точка bbox — приближённо точка контакта человека с землёй."""
    return ((det.x1 + det.x2) // 2, det.y2)


def denormalize_polygon(points_norm: list[list[float]],
                        width: int, height: int) -> list[tuple[int, int]]:
    """Перевод нормированных координат [0..1] в пиксели кадра."""
    return [(int(x * width), int(y * height)) for x, y in points_norm]


def point_in_polygon(point: tuple[int, int],
                     polygon_px: list[tuple[int, int]]) -> bool:
    """Проверка попадания точки в полигон (включая границу)."""
    contour = np.array(polygon_px, dtype=np.int32)
    return cv2.pointPolygonTest(contour, point, False) >= 0


def draw_polygon(frame, polygon_px: list[tuple[int, int]],
                 color=(0, 165, 255), label: str = "ROI") -> None:
    """Рисует полигон ROI и подпись."""
    pts = np.array(polygon_px, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
    if polygon_px:
        cv2.putText(frame, label, (polygon_px[0][0], polygon_px[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
