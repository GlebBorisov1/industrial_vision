from __future__ import annotations
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2

from src.events.event_engine import Event

logger = logging.getLogger(__name__)


class EventStorage:
    """
    Сохраняет события на диск в структуре:
        <base_dir>/YYYY/MM/DD/<event_uuid>/event.json
        <base_dir>/YYYY/MM/DD/<event_uuid>/snapshot.jpg
    """

    def __init__(self, base_dir: str = "events"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"EventStorage инициализирован: {self.base_dir.resolve()}")

    def _event_dir(self, event: Event) -> Path:
        # timestamp в ISO-формате: 2026-11-06T15:44:54.506577+00:00
        ts = datetime.fromisoformat(event.timestamp)
        return self.base_dir / f"{ts:%Y}" / f"{ts:%m}" / f"{ts:%d}" / event.event_id

    def save(self, event: Event, frame=None) -> Path:
        """Сохраняет event.json и (опционально) snapshot.jpg. Возвращает путь к папке события."""
        event_dir = self._event_dir(event)
        event_dir.mkdir(parents=True, exist_ok=True)

        snapshot_rel: str | None = None
        if frame is not None:
            snapshot_path = event_dir / "snapshot.jpg"
            ok = cv2.imwrite(str(snapshot_path), frame)
            if ok:
                # Относительный путь от корня проекта
                snapshot_rel = str(snapshot_path.as_posix())
                event.snapshot = snapshot_rel
                logger.debug(f"Snapshot сохранён: {snapshot_rel}")
            else:
                logger.error(f"Не удалось сохранить snapshot: {snapshot_path}")

        # Записываем event.json (уже с полем snapshot, если оно было)
        json_path = event_dir / "event.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(event), f, ensure_ascii=False, indent=2)

        logger.info(f"[STORAGE] {event.event_type} → {event_dir}")
        return event_dir

    def list_events(self, limit: int | None = None) -> list[dict]:
        """
        Возвращает список всех событий (по убыванию timestamp).
        Пригодится для REST API на Этапе 9.
        """
        events = []
        for json_path in sorted(self.base_dir.rglob("event.json"), reverse=True):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    events.append(json.load(f))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Не удалось прочитать {json_path}: {e}")
        return events[:limit] if limit else events
