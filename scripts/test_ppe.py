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
from src.analytics.ppe_checker import check_ppe_for_person
from src.events.event_engine import EventEngine
import logging


def draw_person_with_ppe(frame, person, result):
    if not result.eligible:
        # Слишком маленький — рисуем серым, не анализируем
        cv2.rectangle(frame, (person.x1, person.y1), (person.x2, person.y2),
                      (128, 128, 128), 1)
        cv2.putText(frame, f"ID {person.track_id} (small)", (person.x1, person.y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1, cv2.LINE_AA)
        return

    # Цвет bbox: красный если любое нарушение, зелёный если всё OK, жёлтый если неизвестно
    violations = []
    if result.has_helmet is False:
        violations.append("NO HELMET")
    if result.has_vest is False:
        violations.append("NO VEST")

    if violations:
        color = (0, 0, 255)         # красный
    elif result.has_helmet is None:
        color = (0, 255, 255)       # жёлтый (неизвестно)
    else:
        color = (0, 255, 0)         # зелёный

    cv2.rectangle(frame, (person.x1, person.y1), (person.x2, person.y2), color, 2)

    # Подпись: ID + статус PPE
    helmet_status = ("H" if result.has_helmet else
                     ("h?" if result.has_helmet is None else "X"))
    vest_status = "V" if result.has_vest else "X"
    label = f"ID {person.track_id} | {helmet_status}{vest_status}"
    if violations:
        label += " " + ",".join(violations)
    cv2.putText(frame, label, (person.x1, person.y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def main():
    setup_logger()
    logger = logging.getLogger("test_ppe")

    with open("configs/configuration.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cam = config["camera"]
    mod = config["models"]
    ppe_cfg = config["ppe"]
    ev_cfg = config["events"]

    video = VideoSource(source=cam["source"], reconnect_sec=cam["reconnect_sec"])

    # Baseline: люди/транспорт + трекинг
    detector = Detector(model_path=mod["detector_path"],
                        confidence=mod["person_confidence"])
    tracker = Tracker(model=detector.model,
                      confidence=mod["person_confidence"],
                      tracker_cfg="configs/bytetrack_custom.yaml")

    # PPE: каска/голова/жилет (без трекинга — используем Track ID от person)
    ppe_detector = Detector(model_path=mod["ppe_model_path"],
                            confidence=mod["ppe_confidence"])

    # Event Engine с конфигами всех событий
    event_configs = {
        "person_in_restricted_zone": ev_cfg["person_in_zone"],
        "ppe_helmet_missing": ev_cfg["ppe_helmet_missing"],
        "ppe_vest_missing": ev_cfg["ppe_vest_missing"],
    }
    engine = EventEngine(camera_id=cam["id"], configs=event_configs)

    fps_meter = FPSMeter(window=30)
    video.open()

    window = "PPE Control - press Q to quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    events_log = []
    frame_no = 0

    PPE_SKIP = int(mod.get("ppe_skip", 2))
    ppe_cache: list = None     # кэш последних PPE-детекций
    try:
        while True:
            frame = video.read()
            if frame is None:
                if video.is_file:
                    video.rewind()
                    engine = EventEngine(camera_id=cam["id"], configs=event_configs)
                    frame_no = 0
                    ppe_cache = None
                    continue
                break

            frame_no += 1

            # 1. Tracking людей
            person_dets = [d for d in tracker.track(frame)
                           if d.class_name == "person" and d.track_id is not None]

            # 2. PPE-детекции (без tracking — одна модель на кадр)
            if ppe_cache is None or frame_no % PPE_SKIP == 0:
                ppe_dets = ppe_detector.predict(frame)
                ppe_cache = ppe_dets
            else:
                ppe_dets = ppe_cache

            # 3. Проверка СИЗ для каждого человека
            active_ids = set()
            for person in person_dets:
                active_ids.add(person.track_id)
                result = check_ppe_for_person(
                    person, ppe_dets,
                    min_person_height=ppe_cfg["min_person_height"],
                    bbox_padding=ppe_cfg["bbox_padding"],
                )

                draw_person_with_ppe(frame, person, result)

                if not result.eligible:
                    continue

                # Каска: событие, если has_helmet is False
                if result.has_helmet is False:
                    ev = engine.update(
                        "ppe_helmet_missing",
                        condition_met=True,
                        track_id=person.track_id,
                        confidence=result.helmet_confidence,
                        coordinates={"x": (person.x1 + person.x2) // 2, "y": person.y2},
                    )
                    if ev: events_log.append(ev)
                else:
                    engine.update("ppe_helmet_missing", condition_met=False,
                                  track_id=person.track_id)

                # Жилет: событие, если has_vest is False
                if result.has_vest is False:
                    ev = engine.update(
                        "ppe_vest_missing",
                        condition_met=True,
                        track_id=person.track_id,
                        confidence=result.vest_confidence,
                        coordinates={"x": (person.x1 + person.x2) // 2, "y": person.y2},
                    )
                    if ev: events_log.append(ev)
                else:
                    engine.update("ppe_vest_missing", condition_met=False,
                                  track_id=person.track_id)

            engine.cleanup(active_ids)

            # HUD
            fps = fps_meter.tick()
            cv2.putText(frame, f"FPS: {fps:5.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"People: {len(person_dets)}  PPE dets: {len(ppe_dets)}",
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
        logger.info(f"  {e.event_type} | track={e.track_id} | conf={e.confidence:.2f}")


if __name__ == "__main__":
    main()
