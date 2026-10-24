# src/capture/video_capture.py
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

    def open(self) -> None:
        if self.cap is not None:
            self.cap.release()

        logger.info(f"Попытка открыть источник видео: {self.source}")
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            logger.error(f"Не удалось открыть источник: {self.source}")
            raise RuntimeError(f"Cannot open video source: {self.source}")

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            self.open()

        ok, frame = self.cap.read()

        if ok and frame is not None:
            return frame

        logger.warning(f"Потеря кадра/потока. Ожидание {self.reconnect_sec} сек...")
        time.sleep(self.reconnect_sec)

        try:
            self.open()
            ok, frame = self.cap.read()
            if not ok:
                raise RuntimeError("Video stream is unavailable after reconnect")
            return frame
        except Exception as e:
            logger.critical(f"Критическая ошибка восстановления потока: {e}")
            raise RuntimeError("Video stream is unavailable after reconnect")

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            logger.info("Источник видео закрыт.")
