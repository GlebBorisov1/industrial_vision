import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_DIR = Path("logs")
MAX_BYTES = 5_000_000      # 5 МБ на файл
BACKUP_COUNT = 5           # 5 ротаций → максимум 30 МБ на журнал

FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def _make_file_handler(path: Path, level: int = logging.INFO) -> RotatingFileHandler:
    path.parent.mkdir(parents=True, exist_ok=True)
    h = RotatingFileHandler(
        path, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    h.setLevel(level)
    h.setFormatter(logging.Formatter(FORMAT))
    return h


def setup_logger() -> None:
    """
    Настраивает 4 независимых журнала с ротацией:
      - application.log     — общий лог приложения (INFO+)
      - ai_events.log       — только события AI (INFO+)
      - camera_errors.log   — ошибки камеры (WARNING+)
      - system_monitor.log  — ресурсы, FPS, latency (INFO+)
    Плюс вывод в stdout для интерактивной разработки.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Не дублируем handlers при повторном вызове
    if any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        return

    # 1. stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(logging.Formatter(FORMAT))
    root.addHandler(stdout_handler)

    # 2. application.log — общий
    root.addHandler(_make_file_handler(LOG_DIR / "application.log"))

    # 3. ai_events.log — только события AI
    ai_logger = logging.getLogger("ai_events")
    ai_logger.setLevel(logging.INFO)
    ai_logger.propagate = False
    ai_logger.addHandler(_make_file_handler(LOG_DIR / "ai_events.log"))

    # 4. camera_errors.log — только ошибки камеры/видео
    cam_logger = logging.getLogger("camera_errors")
    cam_logger.setLevel(logging.WARNING)
    cam_logger.propagate = False
    cam_logger.addHandler(_make_file_handler(LOG_DIR / "camera_errors.log", logging.WARNING))

    # 5. system_monitor.log — метрики
    sys_logger = logging.getLogger("system_monitor")
    sys_logger.setLevel(logging.INFO)
    sys_logger.propagate = False
    sys_logger.addHandler(_make_file_handler(LOG_DIR / "system_monitor.log"))


def get_event_logger() -> logging.Logger:
    """Логгер для записи AI-событий в отдельный файл."""
    return logging.getLogger("ai_events")


def get_camera_logger() -> logging.Logger:
    """Логгер для ошибок видеопотока."""
    return logging.getLogger("camera_errors")


def get_monitor_logger() -> logging.Logger:
    """Логгер для метрик производительности."""
    return logging.getLogger("system_monitor")
