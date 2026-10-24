# src/capture/fps_meter.py
import time
from collections import deque

class FPSMeter:
    def __init__(self, window: int = 30):
        # Храним время последних 30 кадров
        self.times = deque(maxlen=window)

    def tick(self) -> float:
        now = time.perf_counter()
        self.times.append(now)

        # Если кадров меньше двух, FPS посчитать нельзя
        if len(self.times) < 2:
            return 0.0

        dt = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / dt if dt > 0 else 0.0
