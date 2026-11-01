import sys, os, logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.logging.logger_setup import setup_logger
from ultralytics import YOLO


def main():
    setup_logger()
    logger = logging.getLogger("tune_confidence")

    data_yaml = "configs/dataset.yaml"   # или локальный путь в venv
    model = YOLO("models/ppe_yolo26n_best.pt")

    results_table = []
    for conf in (0.30, 0.50, 0.70):
        logger.info(f"=== Прогон conf={conf} ===")
        m = model.val(data=data_yaml, split="test", conf=conf, verbose=False)
        r = m.results_dict
        p = r.get("metrics/precision(B)", 0)
        rc = r.get("metrics/recall(B)", 0)
        f1 = 2 * p * rc / (p + rc) if (p + rc) else 0
        results_table.append((conf, p, rc, f1))
        logger.info(f"conf={conf}: P={p:.4f} R={rc:.4f} F1={f1:.4f}")

    logger.info("\n=== ИТОГОВАЯ ТАБЛИЦА ===")
    logger.info(f"{'conf':>6} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
    for conf, p, rc, f1 in results_table:
        logger.info(f"{conf:>6.2f} | {p:>10.4f} | {rc:>10.4f} | {f1:>10.4f}")


if __name__ == "__main__":
    main()
