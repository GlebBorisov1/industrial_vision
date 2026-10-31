import sys
import os
import yaml
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.logging.logger_setup import setup_logger
from src.capture.video_capture import VideoSource
from src.capture.fps_meter import FPSMeter
import logging


def main():
    setup_logger()
    logger = logging.getLogger("test_capture")

    # 1. Читаем конфиг
    with open("configs/configuration.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    source_path = config["camera"]["source"]
    reconnect_sec = config["camera"]["reconnect_sec"]

    logger.info(f"Запуск теста. Источник: {source_path}")

    # 2. Инициализация
    video = VideoSource(source=source_path, reconnect_sec=reconnect_sec)
    fps_meter = FPSMeter(window=30)

    try:
        video.open()
    except RuntimeError as e:
        logger.critical(f"Не удалось запустить видео: {e}")
        return

    window_name = "Video Capture - press Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            frame = video.read()

            # Конец файла → зацикливаем для удобства демонстрации
            if frame is None:
                if video.is_file:
                    video.rewind()
                    fps_meter = FPSMeter(window=30)  # сбрасываем окно FPS
                    continue
                else:
                    logger.info("Поток завершён.")
                    break

            # Измеряем FPS обработки
            current_fps = fps_meter.tick()

            # Накладываем FPS на кадр
            cv2.putText(
                frame,
                f"FPS: {current_fps:5.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Source: {'file' if video.is_file else 'stream'}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame)

            # Выход по Q или ESC
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                logger.info("Завершение по запросу пользователя.")
                break

            # Если окно закрыто крестиком — тоже выходим
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                logger.info("Окно закрыто пользователем.")
                break

    except KeyboardInterrupt:
        logger.info("Остановка по Ctrl+C.")
    except Exception as e:
        logger.error(f"Ошибка во время чтения: {e}")
    finally:
        video.close()
        cv2.destroyAllWindows()
        logger.info("Тест завершён.")


if __name__ == "__main__":
    main()
