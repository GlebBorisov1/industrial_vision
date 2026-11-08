import sys
import os
import yaml
import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.logging.logger_setup import setup_logger, get_event_logger
from src.capture.video_capture import VideoSource
from src.capture.fps_meter import FPSMeter
from src.events.event_engine import EventEngine
from src.events.storage import EventStorage
import logging


def main():
    setup_logger()
    logger = logging.getLogger("test_spill_seg")

    with open("configs/configuration.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cam = config["camera"]
    spill_cfg = config.get("spill", {})
    ev_cfg = config["events"]["spill_detected"]

    logger.info("Загрузка модели сегментации разлива...")
    weights_path = hf_hub_download(
        repo_id="kareem1232/spill_leak_detection_nano",
        filename="weights/best.pt",
        local_dir="models/spill_seg"
    )
    logger.info(f"Модель загружена: {weights_path}")

    # 2. Инициализация модели YOLO (задача segment определяется автоматически)
    model = YOLO(weights_path)
    print(model.names)

    # Проверяем, что модель действительно сегментационная
    logger.info(f"Задача модели: {model.task}")
    if model.task != "segment":
        logger.warning("Модель не является сегментационной! Маски могут отсутствовать.")

    # 3. Настройка видео и Event Engine
    video = VideoSource(source=cam["source"], reconnect_sec=cam["reconnect_sec"])

    event_configs = {"spill_detected": ev_cfg}

    def make_engine() -> EventEngine:
        return EventEngine(camera_id=cam["id"], configs=event_configs)

    engine = make_engine()
    fps_meter = FPSMeter(window=30)
    storage = EventStorage(base_dir="events")
    event_log = get_event_logger()

    video.open()
    window = "Spill Segmentation - press Q to quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    events_log = []
    # Порог уверенности для события (можно вынести в конфиг)
    conf_threshold = spill_cfg.get("confidence", 0.5)
    # Минимальная площадь маски в пикселях (фильтр мелких брызг)
    min_mask_area = spill_cfg.get("min_area_px", 3000)

    try:
        while True:
            frame = video.read()
            if frame is None:
                if video.is_file:
                    video.rewind()
                    engine = make_engine()
                    continue
                break

            # 4. Инференс сегментации
            #    conf — порог уверенности, iou — порог NMS, verbose=False чтобы не спамить
            results = model.predict(frame, conf=conf_threshold, iou=0.5, verbose=False)
            result = results[0]

            # 5. Извлечение масок (если они есть)
            spill_present = False
            best_conf = 0.0
            total_mask_area = 0
            masks_data = None

            if result.masks is not None and len(result.masks) > 0:
                # masks.data — тензор (N, H, W) с бинарными масками
                # Приводим к numpy и обрезаем до размера исходного кадра
                masks_data = result.masks.data.cpu().numpy()  # (N, H, W)
                h, w = frame.shape[:2]
                # Некоторые модели возвращают маски с другим разрешением — ресайзим
                if masks_data.shape[1:] != (h, w):
                    masks_resized = np.zeros((masks_data.shape[0], h, w), dtype=np.uint8)
                    for i, m in enumerate(masks_data):
                        masks_resized[i] = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    masks_data = masks_resized

                # Считаем площадь каждой маски и оставляем только крупные
                for i, mask in enumerate(masks_data):
                    area = int(np.sum(mask))
                    if area < min_mask_area:
                        continue

                    conf = float(result.boxes.conf[i]) if result.boxes is not None else 0.0
                    spill_present = True
                    best_conf = max(best_conf, conf)
                    total_mask_area += area

                    # Визуализация маски: накладываем цветной слой
                    color_mask = np.zeros_like(frame)
                    color_mask[mask > 0] = (0, 165, 255)  # оранжевый
                    frame = cv2.addWeighted(frame, 1.0, color_mask, 0.4, 0)

                    # Рисуем контур маски (по желанию)
                    contours, _ = cv2.findContours(
                        (mask * 255).astype(np.uint8),
                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(frame, contours, -1, (0, 165, 255), 2)

                    # Подпись с площадью и уверенностью
                    x, y, w_box, h_box = cv2.boundingRect(contours[0]) if contours else (0, 0, 0, 0)
                    label = f"spill {conf:.2f} area={area}px²"
                    cv2.putText(frame, label, (x, max(20, y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2, cv2.LINE_AA)

            # 6. Event Engine: условие — есть хотя бы одна уверенная маска
            coords = {}
            if spill_present and result.boxes is not None:
                # Берём центр первой крупной маски (приблизительно)
                b = result.boxes.xyxy[0].cpu().numpy().astype(int)
                coords = {"x": int((b[0] + b[2]) / 2), "y": int((b[1] + b[3]) / 2)}

            event = engine.update(
                "spill_detected", spill_present,
                track_id=None,
                confidence=best_conf,
                coordinates=coords,
                extra={"mask_area_px": total_mask_area} if spill_present else None,
            )

            if event is not None:
                events_log.append(event)
                event_log.info(
                    f"{event.event_type} conf={event.confidence:.2f} area={total_mask_area}px²"
                )
                storage.save(event, frame=frame)

            # HUD
            fps = fps_meter.tick()
            cv2.putText(frame, f"FPS: {fps:5.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Spill: {spill_present}  Area: {total_mask_area}px²",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Events: {len(events_log)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        logger.info("Остановлено пользователем.")
    finally:
        video.close()
        cv2.destroyAllWindows()

    logger.info(f"Сессия завершена. Всего событий: {len(events_log)}")
    for e in events_log:
        logger.info(f"  {e.event_type} | conf={e.confidence:.2f} | {e.timestamp}")


if __name__ == "__main__":
    main()
