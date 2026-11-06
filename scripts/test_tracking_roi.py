import sys
import os
import yaml
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.logging.logger_setup import setup_logger
from src.capture.video_capture import VideoSource
from src.capture.fps_meter import FPSMeter
from src.inference.detector import Detector
from src.tracking.tracker import Tracker
from src.zones.geometry import (bottom_center, denormalize_polygon,
                                point_in_polygon, draw_polygon)
from src.events.event_engine import EventEngine
import logging

EVENT_TYPE = "person_in_restricted_zone"

def main():
    setup_logger()
    logger = logging.getLogger("test_tracking_roi")

    with open("configs/configuration.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cam = config["camera"]
    mod = config["models"]
    zone_cfg = config["zones"]["restricted_01"]
    ev_cfg = config["events"]["person_in_zone"]

    video = VideoSource(source=cam["source"], reconnect_sec=cam["reconnect_sec"])
    detector = Detector(model_path=mod["detector_path"],
                        confidence=mod["person_confidence"])
    tracker = Tracker(
        model=detector.model,
        confidence=mod["person_confidence"],
        tracker_cfg="configs/bytetrack_custom.yaml"
    )
    event_configs = {EVENT_TYPE: ev_cfg}

    def make_engine() -> EventEngine:
        return EventEngine(camera_id=cam["id"], configs=event_configs)

    engine = make_engine()
    fps_meter = FPSMeter(window=30)

    video.open()
    window = "Tracking + ROI - press Q to quit"
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

            h, w = frame.shape[:2]
            polygon_px = denormalize_polygon(zone_cfg["points"], w, h)
            draw_polygon(frame, polygon_px, label="RESTRICTED ZONE")

            # Tracking
            detections = tracker.track(frame)
            active_ids = set()

            for det in detections:
                if det.class_name != "person" or det.track_id is None:
                    continue

                active_ids.add(det.track_id)
                pt = bottom_center(det)
                in_zone = point_in_polygon(pt, polygon_px)

                # Цвет bbox: зелёный если вне зоны, красный если внутри
                color = (0, 0, 255) if in_zone else (0, 255, 0)
                cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, 2)
                cv2.circle(frame, pt, 5, color, -1)

                label = f"ID {det.track_id} {det.confidence:.2f}"
                if in_zone:
                    label += " [IN ZONE]"
                cv2.putText(frame, label, (det.x1, det.y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

                # Event Engine
                event = engine.update(
                    EVENT_TYPE,
                    in_zone,
                    track_id=det.track_id,
                    confidence=det.confidence,
                    coordinates={"x": int(pt[0]), "y": int(pt[1])},
                )
                if event is not None:
                    events_log.append(event)

            engine.cleanup(active_ids)

            # FPS + счётчик событий
            fps = fps_meter.tick()
            cv2.putText(frame, f"FPS: {fps:5.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Events: {len(events_log)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

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
        logger.info(f"  {e.event_type} | track={e.track_id} | "
                    f"conf={e.confidence:.2f} | {e.timestamp}")


if __name__ == "__main__":
    main()
