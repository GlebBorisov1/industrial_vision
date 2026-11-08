import sys
import os
import yaml
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.logging.logger_setup import setup_logger, get_event_logger
from src.capture.video_capture import VideoSource
from src.capture.fps_meter import FPSMeter
from src.analytics.spill_detector import SpillDetector
from src.events.event_engine import EventEngine
from src.events.storage import EventStorage
import logging

EVENT_TYPE = "spill_detected"


def main():
    setup_logger()
    logger = logging.getLogger("test_spill")

    with open("configs/configuration.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cam = config["camera"]
    spill_cfg = config["spill"]
    ev_cfg = config["events"]["spill_detected"]

    video = VideoSource(source=cam["source"], reconnect_sec=cam["reconnect_sec"])

    spill_detector = SpillDetector(
        min_area_px=spill_cfg["min_area_px"],
        max_aspect_ratio=spill_cfg["max_aspect_ratio"],
        hsv_low=tuple(spill_cfg["hsv_low"]),
        hsv_high=tuple(spill_cfg["hsv_high"]),
    )

    event_configs = {EVENT_TYPE: ev_cfg}

    def make_engine() -> EventEngine:
        return EventEngine(camera_id=cam["id"], configs=event_configs)

    engine = make_engine()
    fps_meter = FPSMeter(window=30)
    storage = EventStorage(base_dir="events")
    event_log = get_event_logger()

    video.open()
    window = "Spill Detection (HSV heuristic) - press Q to quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    events_log = []

    try:
        while True:
            frame = video.read()
            if frame is None:
                if video.is_file:
                    video.rewind()
                    engine = make_engine()
                    continue
                break

            # 1. Детекция разлива (эвристика — очень быстро)
            spill_dets = spill_detector.detect(frame)

            # 2. Фильтрация по порогу confidence
            confident = [d for d in spill_dets
                         if d.confidence >= spill_cfg["confidence"]]

            # 3. Визуализация
            for det in spill_dets:
                passed = det.confidence >= spill_cfg["confidence"]
                color = (0, 165, 255) if passed else (128, 128, 128)  # оранжевый/серый
                cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, 2)
                label = f"spill {det.confidence:.2f}"
                if not passed:
                    label += " [low conf]"
                cv2.putText(frame, label, (det.x1, det.y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            # 4. Event Engine — условие = есть хотя бы одна уверенная детекция
            spill_present = len(confident) > 0
            best_conf = max((d.confidence for d in confident), default=0.0)
            best_bbox = confident[0] if confident else None

            coords = {}
            if best_bbox is not None:
                coords = {
                    "x": (best_bbox.x1 + best_bbox.x2) // 2,
                    "y": (best_bbox.y1 + best_bbox.y2) // 2,
                }

            event = engine.update(
                EVENT_TYPE, spill_present,
                track_id=None,
                confidence=best_conf,
                coordinates=coords,
                extra={"detections_count": len(confident)} if spill_present else None,
            )

            if event is not None:
                events_log.append(event)
                event_log.info(
                    f"{event.event_type} conf={event.confidence:.2f} "
                    f"dets={len(confident)}"
                )
                storage.save(event, frame=frame)

            # HUD
            fps = fps_meter.tick()
            cv2.putText(frame, f"FPS: {fps:5.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Spill: {spill_present}  Dets: {len(spill_dets)}",
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
