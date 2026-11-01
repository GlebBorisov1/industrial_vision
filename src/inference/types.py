from dataclasses import dataclass
from typing import Optional

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    track_id: Optional[int] = None
