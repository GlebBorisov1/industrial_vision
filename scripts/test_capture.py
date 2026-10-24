# scripts/test_capture.py
import sys
import os
import yaml
import cv2

# Добавляем корневую папку в sys.path, чтобы работали импорты из src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.logging.logger_setup import setup_logger
from src.capture.video_capture import VideoSource
from src.capture.fps_meter import FPSMeter
import logging

def main():
    setup_logger()
    logger = logging.getLogger("test_capture")

    # 1. Читаем конфиг
    config_path = "configs/configuration.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    source_path = config["camera"]["source"]
    reconnect_sec = config["camera"]["reconnect_sec"]

    logger.info(f"Запуск теста. Источник: {source_path}")

    # 2. Инициализируем классы
    video = VideoSource(source=source_path, reconnect_sec=reconnect_sec)
    fps_meter = FPSMeter(window=30)

    try:
        video.open()
    except RuntimeError as e:
        logger.critical(f"Не удалось запустить видео: {e}")
        return

    frame_count = 0

    # 3. Цикл обработки
    try:
        while True:
            frame = video.read()
            current_fps = fps_meter.tick()
            frame_count += 1

            # Раз в 30 кадров выводим статистику в консоль
            if frame_count % 30 == 0:
                logger.info(f"Обработано кадров: {frame_count}. Текущий FPS: {current_fps:.2f}")

            # Сохраняем один кадр для визуальной проверки
            if frame_count == 10:
                cv2.imwrite("data/test_frame.jpg", frame)
                logger.info("Тестовый кадр сохранен в data/test_frame.jpg")

            # Ограничиваем цикл для теста (например, 300 кадрами, чтобы не висело вечно)
            if frame_count >= 300:
                logger.info("Достигнут лимит кадров (300). Завершение теста.")
                break

    except KeyboardInterrupt:
        logger.info("Остановка по команде пользователя.")
    except Exception as e:
        logger.error(f"Произошла ошибка во время чтения: {e}")
    finally:
        video.close()

if __name__ == "__main__":
    main()
