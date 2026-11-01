from __future__ import annotations
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class Event:
    event_id: str
    event_type: str
    timestamp: str
    camera_id: str
    confidence: float
    track_id: int | None
    coordinates: dict
    snapshot: str | None = None
    status: str = "new"


class TemporalConfirm:
    """N из M: событие подтверждается, если условие выполнено >= N раз в последних M кадрах."""

    def __init__(self, n: int, m: int):
        if n > m:
            raise ValueError("n не может быть больше m")
        self.n = n
        self.window: deque[bool] = deque(maxlen=m)

    def update(self, detected: bool) -> bool:
        self.window.append(bool(detected))
        # Событие формируется только когда окно заполнено
        return len(self.window) == self.window.maxlen and sum(self.window) >= self.n

    def reset(self) -> None:
        self.window.clear()


class EventEngine:
    def __init__(self, camera_id: str, temporal_n: int = 5, temporal_m: int = 8,
                 cooldown_sec: float = 10.0):
        self.camera_id = camera_id
        self.temporal_n = temporal_n
        self.temporal_m = temporal_m
        self.cooldown_sec = cooldown_sec
        # Храним состояние по каждому Track ID
        self._confirmers: dict[int, TemporalConfirm] = {}
        self._last_event_ts: dict[int, float] = {}
        self._active_zone: dict[int, bool] = {}

    def _get_confirmer(self, track_id: int) -> TemporalConfirm:
        if track_id not in self._confirmers:
            self._confirmers[track_id] = TemporalConfirm(self.temporal_n, self.temporal_m)
        return self._confirmers[track_id]

    def _in_cooldown(self, track_id: int) -> bool:
        last = self._last_event_ts.get(track_id)
        return last is not None and (time.time() - last) < self.cooldown_sec

    def update(self, track_id: int, in_zone: bool, confidence: float,
               point: tuple[int, int]) -> Event | None:
        """
        Вызывается на каждом кадре для каждого объекта person в зоне.
        Возвращает Event, если он должен быть сгенерирован.
        """
        confirmer = self._get_confirmer(track_id)
        confirmed = confirmer.update(in_zone)

        # Событие только по фронту (переход False → True) + cooldown
        if confirmed and not self._active_zone.get(track_id, False):
            self._active_zone[track_id] = True
            if self._in_cooldown(track_id):
                logger.debug(f"Track {track_id}: cooldown, событие подавлено.")
                return None

            event = Event(
                event_id=str(uuid4()),
                event_type="person_in_restricted_zone",
                timestamp=datetime.now(timezone.utc).isoformat(),
                camera_id=self.camera_id,
                confidence=float(confidence),
                track_id=track_id,
                coordinates={"x": int(point[0]), "y": int(point[1])},
            )
            self._last_event_ts[track_id] = time.time()
            logger.info(f"[EVENT] {event.event_type} track_id={track_id} "
                        f"conf={confidence:.2f} at {point}")
            return event

        # Объект вышел из зоны — сбрасываем состояние
        if not in_zone:
            self._active_zone[track_id] = False
            confirmer.reset()

        return None

    def cleanup(self, active_track_ids: set[int]) -> None:
        """Удаляет состояния для треков, которые пропали из кадра."""
        for tid in list(self._confirmers.keys()):
            if tid not in active_track_ids:
                del self._confirmers[tid]
                self._active_zone.pop(tid, None)
