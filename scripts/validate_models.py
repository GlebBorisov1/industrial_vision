import sys, os, yaml, logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.logging.logger_setup import setup_logger
from ultralytics import YOLO

def main():
    setup_logger()
    logger = logging.getLogger("validate")

    data_yaml = "/app/datasets/ppe_test_only/data.yaml"
    weights_ppe = "models/ppe_yolo26n_best.pt"
    weights_baseline = "models/yolov8n.pt"  # скачается автоматически (COCO)

    logger.info("=== Валидация YOLO26n (PPE) на test split ===")
    model_26 = YOLO(weights_ppe)
    metrics_26 = model_26.val(data=data_yaml, split="test", verbose=False)

    # Красивый вывод метрик
    r = metrics_26.results_dict
    logger.info(f"YOLO26n PPE Precision:  {r.get('metrics/precision(B)', 0):.4f}")
    logger.info(f"YOLO26n PPE Recall:     {r.get('metrics/recall(B)', 0):.4f}")
    logger.info(f"YOLO26n PPE mAP@0.5:    {r.get('metrics/mAP50(B)', 0):.4f}")
    logger.info(f"YOLO26n PPE mAP@0.5:0.95: {r.get('metrics/mAP50-95(B)', 0):.4f}")
    logger.info(f"Полный словарь метрик: {r}")

    # Скорость инференса
    speed = metrics_26.speed  # dict: preprocess, inference, loss, postprocess (ms/image)
    logger.info(f"Скорость YOLO26n: {speed}")

if __name__ == "__main__":
    main()
