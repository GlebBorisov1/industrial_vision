from __future__ import annotations
import time
import logging
from collections import deque
from dataclasses import dataclass
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
    extra: dict | None = None


class TemporalConfirm:
    """N из M подтверждений в скользящем окне."""

    def __init__(self, n: int, m: int):
        if n > m:
            raise ValueError("n не может быть больше m")
        self.n = n
        self.m = m
        self.window: deque[bool] = deque(maxlen=m)

    def update(self, detected: bool) -> bool:
        self.window.append(bool(detected))
        return len(self.window) == self.m and sum(self.window) >= self.n

    def reset(self) -> None:
        self.window.clear()


class EventEngine:
    """
    Универсальный движок событий.
    Хранит состояние (confirmer + cooldown + active flag) по ключу (track_id, event_type).
    """

    def __init__(self, camera_id: str,
                 configs: dict[str, dict]):
        """
        :param camera_id: идентификатор камеры
        :param configs: словарь {event_type: {"temporal_n", "temporal_m", "cooldown_sec"}}
        """
        self.camera_id = camera_id
        self.configs = configs
        self._confirmers: dict[tuple[int | None, str], TemporalConfirm] = {}
        self._last_event_ts: dict[tuple[int | None, str], float] = {}
        self._active: dict[tuple[int | None, str], bool] = {}

    def _conf(self, event_type: str) -> dict:
        if event_type not in self.configs:
            raise KeyError(f"Нет конфигурации для события '{event_type}'")
        return self.configs[event_type]

    def _get_confirmer(self, key: tuple[int | None, str]) -> TemporalConfirm:
        if key not in self._confirmers:
            c = self._conf(key[1])
            self._confirmers[key] = TemporalConfirm(c["temporal_n"], c["temporal_m"])
        return self._confirmers[key]

    def _in_cooldown(self, key: tuple[int | None, str]) -> bool:
        last = self._last_event_ts.get(key)
        if last is None:
            return False
        return (time.time() - last) < self._conf(key[1])["cooldown_sec"]

    def update(self, event_type: str, condition_met: bool, *,
               track_id: int | None = None,
               confidence: float = 0.0,
               coordinates: dict | None = None,
               extra: dict | None = None) -> Event | None:
        """
        Вызывается каждый кадр для каждого объекта/условия.
        Возвращает Event по фронту (False → True) после temporal confirmation и с учётом cooldown.
        """
        key = (track_id, event_type)
        confirmer = self._get_confirmer(key)
        confirmed = confirmer.update(condition_met)

        if confirmed and not self._active.get(key, False):
            self._active[key] = True
            if self._in_cooldown(key):
                logger.debug(f"Cooldown для {event_type} track={track_id} — событие подавлено.")
                return None

            event = Event(
                event_id=str(uuid4()),
                event_type=event_type,
                timestamp=datetime.now(timezone.utc).isoformat(),
                camera_id=self.camera_id,
                confidence=float(confidence),
                track_id=track_id,
                coordinates=coordinates or {},
                extra=extra,
            )
            self._last_event_ts[key] = time.time()
            logger.info(f"[EVENT] {event_type} track={track_id} conf={confidence:.2f}")
            return event

        # Условие больше не выполняется — сбрасываем фронт
        if not condition_met:
            self._active[key] = False
            confirmer.reset()

        return None

    def cleanup(self, active_track_ids: set[int]) -> None:
        """Удаляет состояние для треков, которых нет в кадре."""
        for key in list(self._confirmers.keys()):
            track_id, _ = key
            if track_id is not None and track_id not in active_track_ids:
                del self._confirmers[key]
                self._active.pop(key, None)
                # _last_event_ts оставляем — cooldown должен пережить короткое исчезновение
