from __future__ import annotations
import time
import logging
import cv2

logger = logging.getLogger(__name__)


class VideoSource:
    def __init__(self, source: str | int, reconnect_sec: float = 2.0):
        self.source = source
        self.reconnect_sec = reconnect_sec
        self.cap: cv2.VideoCapture | None = None
        # Определяем: это файл или живой поток (камера / RTSP / HTTP)?
        self.is_file = isinstance(source, str) and not any(
            source.startswith(prefix) for prefix in ("rtsp://", "rtmp://", "http://", "https://")
        )
        logger.info(f"Источник определён как {'файл' if self.is_file else 'поток'}: {source}")

    def open(self) -> None:
        if self.cap is not None:
            self.cap.release()

        logger.info(f"Попытка открыть источник видео: {self.source}")
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            logger.error(f"Не удалось открыть источник: {self.source}")
            raise RuntimeError(f"Cannot open video source: {self.source}")

    def read(self):
        """Возвращает кадр или None в конце файла. Для потоков при обрыве пытается переподключиться."""
        if self.cap is None or not self.cap.isOpened():
            self.open()

        ok, frame = self.cap.read()

        if ok and frame is not None:
            return frame

        # Конец файла — это НЕ ошибка, возвращаем None
        if self.is_file:
            logger.info("Достигнут конец видеофайла.")
            return None

        # Для потока (камера / RTSP) — пытаемся переподключиться
        logger.warning(f"Потеря потока. Ожидание {self.reconnect_sec} сек...")
        time.sleep(self.reconnect_sec)

        try:
            self.open()
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError("Video stream is unavailable after reconnect")
            logger.info("Поток успешно восстановлен.")
            return frame
        except Exception as e:
            logger.critical(f"Критическая ошибка восстановления потока: {e}")
            raise RuntimeError("Video stream is unavailable after reconnect")

    def rewind(self) -> None:
        """Возвращает видеофайл в начало (только для файлов)."""
        if self.is_file and self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            logger.info("Видеофайл возвращён в начало.")

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            logger.info("Источник видео закрыт.")
