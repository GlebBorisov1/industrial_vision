import sys
import os
import time
import yaml
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.logging.logger_setup import setup_logger
from src.logging.logger_setup import get_event_logger
from src.capture.video_capture import VideoSource
from src.capture.fps_meter import FPSMeter
from src.inference.detector import Detector
from src.analytics.smoke_fire_filter import filter_smoke_fire
from src.events.event_engine import EventEngine
from src.events.storage import EventStorage
import logging


def main():
    setup_logger()
    logger = logging.getLogger("test_smoke_fire")

    with open("configs/configuration.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cam = config["camera"]
    mod = config["models"]
    sf_cfg = config["smoke_fire"]
    ev_cfg = config["events"]

    video = VideoSource(source=cam["source"], reconnect_sec=cam["reconnect_sec"])
    detector = Detector(model_path=mod["fire_model_path"],
                        confidence=min(mod["fire_confidence"], mod["smoke_confidence"]))

    event_configs = {
        "smoke_detected": ev_cfg["smoke_detected"],
        "fire_detected": ev_cfg["fire_detected"],
    }
    engine = EventEngine(camera_id=cam["id"], configs=event_configs)
    fps_meter = FPSMeter(window=30)

    video.open()
    window = "Smoke/Fire Detection - press Q to quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    events_log = []
    # Для измерения latency: словарь "первое появление признака" -> timestamp
    first_seen: dict[str, float] = {}
    storage = EventStorage(base_dir="events")
    event_log = get_event_logger()

    try:
        while True:
            frame = video.read()
            if frame is None:
                if video.is_file:
                    video.rewind()
                    engine = EventEngine(camera_id=cam["id"], configs=event_configs)
                    first_seen.clear()
                    continue
                break

            t_frame_start = time.perf_counter()

            # 1. Детекция (обе модели уже в одном Detector.predict)
            raw_dets = detector.predict(frame)

            # 2. Фильтрация по hard negatives
            filtered = filter_smoke_fire(
                raw_dets,
                min_area_px=sf_cfg["min_area_px"],
                max_aspect_ratio=sf_cfg["max_aspect_ratio"],
            )

            # 3. Проверка условий для событий
            smoke_present = False
            fire_present = False
            best_smoke_conf = 0.0
            best_fire_conf = 0.0

            for f in filtered:
                det = f.det
                color = (0, 255, 0) if f.passed else (128, 128, 128)
                cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, 2)
                label = f"{det.class_name} {det.confidence:.2f}"
                if not f.passed:
                    label += f" [{f.reason.split('(')[0].strip()}]"
                cv2.putText(frame, label, (det.x1, det.y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

                if not f.passed:
                    continue

                if det.class_name == "smoke" and det.confidence >= mod["smoke_confidence"]:
                    smoke_present = True
                    best_smoke_conf = max(best_smoke_conf, det.confidence)
                elif det.class_name == "fire" and det.confidence >= mod["fire_confidence"]:
                    fire_present = True
                    best_fire_conf = max(best_fire_conf, det.confidence)

            # 4. Отслеживание времени первого появления (для latency)
            now = time.perf_counter()
            if smoke_present and "smoke_detected" not in first_seen:
                first_seen["smoke_detected"] = now
            if fire_present and "fire_detected" not in first_seen:
                first_seen["fire_detected"] = now

            # 5. Event Engine
            smoke_event = engine.update("smoke_detected", smoke_present,
                                        track_id=None, confidence=best_smoke_conf,
                                        coordinates={"source": "scene"})
            fire_event = engine.update("fire_detected", fire_present,
                                       track_id=None, confidence=best_fire_conf,
                                       coordinates={"source": "scene"})

            for ev in (smoke_event, fire_event):
                if ev is None:
                    continue
                events_log.append(ev)
                latency = now - first_seen.get(ev.event_type, now)
                logger.info(f"[EVENT] {ev.event_type} conf={ev.confidence:.2f} "
                            f"latency={latency:.2f}s")
                event_log.info(
                    f"{ev.event_type} conf={ev.confidence:.2f} latency={latency:.2f}s"
                )
                ev.extra = {"latency_sec": round(latency, 3)}
                storage.save(ev, frame=frame)

            # HUD
            fps = fps_meter.tick()
            cv2.putText(frame, f"FPS: {fps:5.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Smoke: {smoke_present}  Fire: {fire_present}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Events: {len(events_log)}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        video.close()
        cv2.destroyAllWindows()

    logger.info(f"Сессия завершена. Всего событий: {len(events_log)}")
    for e in events_log:
        logger.info(f"  {e.event_type} | conf={e.confidence:.2f} | {e.timestamp}")


if __name__ == "__main__":
    main()
