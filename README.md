# Модуль промышленного зрения

Разрабатываемый модуль предназначен для автоматизированного анализа видеопотока, получаемого от камеры мобильной роботизированной платформы, выполняющей инспекционный обход промышленной территории. Основная обработка изображений производится локально на бортовом вычислительном устройстве, что позволяет снизить зависимость от внешнего канала связи и уменьшить задержку реакции на события.

Модуль должен принимать видеопоток, извлекать кадры, выполнять предварительную обработку, запускать модели компьютерного зрения, отслеживать обнаруженные объекты, анализировать контролируемые области изображения, формировать события, сохранять подтверждающие изображения, вести журнал и предоставлять результаты внешнему программному интерфейсу.

## 📋 Назначение

Модуль предназначен для:

- Получения видеопотока (файл, USB-камера, RTSP-поток).
- Детекции людей и транспорта (baseline-модель).
- Контроля средств индивидуальной защиты (СИЗ): каска и жилет.
- Детекция дыма и огня — модель fire-smoke (YOLOv8n), с фильтрацией hard negatives.
- Tracking объектов (ByteTrack) с устойчивыми Track ID при окклюзиях.
- ROI-аналитика: полигон запретной зоны в нормированных координатах.
- Event Engine: универсальный движок событий с temporal confirmation и cooldown.
- Генерация событий `person_in_restricted_zone`, `ppe_helmet_missing`, `ppe_vest_missing`, `smoke_detected`, `fire_detected`.
- Оптимизация инференса через OpenVINO для Edge-инференса на CPU.

## 🏗 Архитектура

Проект построен по модульному принципу. Основные слои:

`Camera` → `VideoCapture` → `Preprocessing` → `AI Inference` → `Tracking` → `ROI/Analytics` → `Event Engine` → `Storage` → `REST API`

Подробная схема и назначение компонентов — в [`docs/architecture.md`](docs/architecture.md).

## 🛠 Технологический стек

| Компонент         | Технология                     |
| ----------------- | ------------------------------ |
| Язык              | Python 3.10                    |
| Контейнеризация   | Docker + Docker Compose        |
| Язык (локальный venv) | Python 3.10+ (проверено на 3.12) |
| Детекция объектов | Ultralytics (YOLOv8n, YOLO26n) |
| Детекция дыма/огня | fire-smoke YOLOv8n (rabahdev) |
| Оптимизация инференса | OpenVINO (FP16, `imgsz=480`) |
| Tracking | ByteTrack |
| Обработка видео   | OpenCV                         |
| REST API          | FastAPI + Uvicorn              |
| Тестирование      | pytest                         |
| Конфигурация      | YAML                           |

## 📁 Структура проекта

```bash
industrial-vision/
├── configs/                       # Конфигурационные файлы (без секретов)
│   ├── configuration.yaml         # Runtime-конфиг (камера, модели, зоны)
│   ├── dataset.yaml               # Документация структуры PPE-датасета
│   ├── bytetrack_custom.yaml # Трекер: ByteTrack, match_thresh=1.0
│   └── botsort_custom.yaml # Трекер: BoT-SORT (альтернатива для сравнения)
├── docs/                          # Архитектура, Dataset Card, Model Card
│   ├── architecture.md            # Архитектура приложения
│   ├── dataset_card.md            # Карта набора данных
│   └── model_card.md              # Карта модели
├── events/
├── logs/
├── models/                        # Веса моделей (не коммитятся, скачиваются отдельно)
│   ├── ppe_yolo26n_best.pt
│   └── yolov8n.pt
├── scripts/                       # Утилиты запуска и проверок
│   ├── benchmark_speed.py
│   ├── debug_fire_model.py        # Диагностика fire/smoke модели
│   ├── test_capture.py
│   ├── test_detector.py
│   ├── test_tracking_roi.py       # Демонстрация Tracking + ROI + Event Engine
│   ├── test_ppe.py                # Контроль СИЗ (helmet/vest)
│   ├── test_smoke_fire.py         # Дым и огонь
│   ├── tune_confidence.py         # Подбор порога (3 прогона val)
│   └── validate_models.py
├── src/                           # Исходный код
│   ├── analytics/                 # Аналитика
│   │   ├── ppe_checker.py         # Логика проверки СИЗ
│   │   └── smoke_fire_filter.py   # Фильтрация hard negatives
│   ├── api/                       # REST API
│   ├── capture/                   # Захват видео, FPS, reconnect
│   │   ├── fps_meter.py
│   │   └── video_capture.py
│   ├── events/                    # Event Engine
│   │   └── event_engine.py
│   ├── inference/                 # Обёртка над Ultralytics (Detector)
│   │   ├── detector.py
│   │   └── types.py
│   ├── logging/                   # Настройка логирования
│   │   └── logger_setup.py
│   ├── monitoring/                # Мониторинг ресурсов
│   ├── tracking/                  # Трекинг объектов
│   │   └── tracker.py
│   └── zones/                     # Работа с ROI
│       └── geometry.py
├── tests/                         # Unit и integration тесты
├── docker-compose.yml
├── Dockerfile
├── .dockerignore
├── .gitignore
├── README.md
└── requirements.txt
```

## 🚀 Режим разработки: venv vs Docker

Проект поддерживает **два режима запуска**:

### Режим 1: локальный venv (активная разработка)

Используется для отладки, визуальной проверки через `cv2.imshow()` и экспериментов с параметрами (пороги, ROI, трекеры). Требует графической оболочки.

```bash
# 1. Создание и активация venv
python3 -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\Activate.ps1       # Windows

# 2. Установка зависимостей
pip install -r requirements.txt
pip install "ultralytics[export-openvino]"   # для OpenVINO-инференса

# 3. Запуск скриптов напрямую
python scripts/test_ppe.py
```

### Режим 2: Docker (финальная сборка и проверка)
Используется для проверки воспроизводимости окружения на машине проверяющего. **Важно**: окно `cv2.imshow()` в контейнере не откроется (нет графической оболочки), поэтому скрипты с визуализацией запускаются только в venv. В Docker доступны headless-проверки (валидация, бенчмарк, будущий REST API).

```bash
docker compose build
docker compose run --rm vision-app python scripts/validate_models.py
docker compose run --rm vision-app python scripts/benchmark_speed.py
```

## 🧠 Модели

<table><thead><tr><th><span class="">Модель</span></th><th><span class="">Задача</span></th><th><span class="">Классы</span></th><th><span class="">Статус</span></th></tr></thead><tbody><tr><td><strong><span class="">YOLOv8n (COCO)</span></strong></td><td><span class="">Детекция людей и транспорта</span></td><td><code>person</code><span class="">, </span><code>car</code><span class="">, </span><code>truck</code><span class="">, </span><code>bus</code><span class="">, </span><code>motorcycle</code></td><td><span class="">✅ Интегрирована, экспортирована в OpenVINO</span></td></tr><tr><td><strong><span class="">YOLO26n (fine-tuned)</span></strong></td><td><span class="">Контроль СИЗ</span></td><td><code>helmet</code><span class="">, </span><code>head</code><span class="">, </span><code>vest</code></td><td><span class="">✅ Обучена, валидирована, экспортирована в OpenVINO</span></td></tr><tr><td><strong><span class="">fire-smoke YOLOv8n</span></strong></td><td><span class="">Детекция дыма и огня</span></td><td><code>smoke</code><span class="">, </span><code>fire</code></td><td><span class="">✅ Интегрирована (источник: </span><code>rabahdev/fire-smoke-yolov8n</code><span class="">)</span></td></tr></tbody></table>

### Метрики YOLO26n (PPE) — PyTorch vs OpenVINO

**До оптимизации** (PyTorch, imgsz=640, test split 2751 изображение):
<table><thead><tr><th><span class="">Метрика</span></th><th><span class="">Значение</span></th></tr></thead><tbody><tr><td><span class="">Precision</span></td><td><span class="">0.9137</span></td></tr><tr><td><span class="">Recall</span></td><td><span class="">0.8937</span></td></tr><tr><td><span class="">mAP@0.5</span></td><td><span class="">0.9492</span></td></tr><tr><td><span class="">mAP@0.5:0.95</span></td><td><span class="">0.6867</span></td></tr><tr><td><span class="">Inference (CPU)</span></td><td><span class="">78.3 мс</span></td></tr></tbody></table>

**После оптимизации** (OpenVINO, imgsz=480, тот же test split):
<table><thead><tr><th><span class="">Метрика</span></th><th><span class="">Значение</span></th><th><span class="">Δ</span></th></tr></thead><tbody><tr><td><span class="">Precision</span></td><td><span class="">0.9080</span></td><td><span class="">−0.6 п.п.</span></td></tr><tr><td><span class="">Recall</span></td><td><span class="">0.8769</span></td><td><span class="">−1.7 п.п.</span></td></tr><tr><td><span class="">mAP@0.5</span></td><td><span class="">0.9353</span></td><td><span class="">−1.4 п.п.</span></td></tr><tr><td><span class="">mAP@0.5:0.95</span></td><td><span class="">0.6670</span></td><td><span class="">−2.0 п.п.</span></td></tr><tr><td><strong><span class="">Inference (CPU)</span></strong></td><td><strong><span class="">35.6 мс</span></strong></td><td><strong><span class="">−54%</span></strong></td></tr></tbody></table>

**Вывод**: экспорт в OpenVINO + уменьшение `imgsz` до 480 даёт **ускорение в 2.2×** при потере 1.4 п.п. mAP@0.5 (93.5% — всё ещё далеко выше целевого порога 70%). Приемлемый компромисс для Edge-инференса на CPU.

Подробности: [`docs/model_card.md`](docs/model_card.md).

### Где взять веса моделей

- `models/yolov8n.pt` — скачивается автоматически при первом запуске `test_detector.py`.
- `models/ppe_yolo26n_best.pt` — обученная в Google Colab модель. Скачайте файл и положите в `models/`.
- `models/fire_detection/fire-smoke.pt` — модель детекции дыма и огня:
    ```bash
    hf download rabahdev/fire-smoke-yolov8n --local-dir models/fire_detection
    ```
- OpenVINO-версии (`*_openvino_model/`) создаются экспортом:
    ```python
    from ultralytics import YOLO
    YOLO("models/yolov8n.pt").export(format="openvino", imgsz=480)
    YOLO("models/ppe_yolo26n_best.pt").export(format="openvino", imgsz=480)
    ```

## 🎯 Tracking (Этап 5)

Для присвоения стабильных Track ID используется `ByteTrack` с кастомной конфигурацией configs/bytetrack_custom.yaml:

```yaml
tracker_type: bytetrack
track_high_thresh: 0.5
track_low_thresh: 0.1
new_track_thresh: 0.6
track_buffer: 60         # 60 кадров ≈ 6–7 с при 10 FPS
match_thresh: 1.0        # ключевой параметр — разрешает сопоставление после окклюзии
fuse_score: True
```
**Ключевой момент:** при `match_thresh < 1.0` трекер отказывается сопоставлять детекцию с потерянным треком после окклюзии, т.к. стоимость сопоставления достигает 1.0. Установка `match_thresh = 1.0` устраняет проблему ID switch при кратковременных исчезновениях (проверено до 6 секунд при FPS ≈ 10).


## 🦺 Контроль СИЗ (Этап 6)
Реализован гибридный подход с двумя моделями (см. `src/analytics/ppe_checker.py`):

1. YOLOv8n находит людей (с Track ID).

2. YOLO26n находит `helmet` / `head` / `vest` на полном кадре.

3. Для каждого человека проверяется наличие СИЗ в его bbox:
    - **Каска**: есть `helmet` → OK; есть `head` → нарушение; ни того, ни другого → неизвестно (событие не генерируется).
    - **Жилет**: есть `vest` → OK; нет `vest` → нарушение.

### Защита от ложных срабатываний
- **Минимальный размер человека:** `min_person_height: 120 px` — мелкие объекты не анализируются (серая рамка `(small)`).
- **Temporal confirmation:** `ppe_helmet_missing` / `ppe_vest_missing` подтверждаются `N=5` из `M=8` кадров.
- **Cooldown:** 30 секунд между событиями для одного Track ID.
- **Bbox padding:** 10% запаса вокруг person bbox для сопоставления PPE.

### Оптимизация производительности
- **OpenVINO FP16**, imgsz=480 для обеих моделей.
- **PPE-инференс не на каждом кадре:** ppe_skip: 2 в конфиге — модель СИЗ запускается раз в 2 кадра, кэш используется на промежуточных. Не влияет на качество событий благодаря temporal confirmation.
- **Итог:** ~11–12 FPS в полном конвейере (две модели + tracking + event logic) на CPU Intel i5-13420H.

## 🔥 Детекция дыма и огня (Этап 7)

Используется предобученная [rabahdev/fire-smoke-yolov8n](https://huggingface.co/rabahdev/fire-smoke-yolov8n) — YOLOv8n с двумя классами: `{0: 'smoke', 1: 'fire'}`. Fine-tuning не проводился: предобученная модель показала приемлемое качество на тестовом видео, а усилия были направлены на интеграцию и защиту от ложных срабатываний.

### Пороги уверенности
По результатам диагностики (`scripts/debug_fire_model.py`) пороги снижены относительно стандартных:

<table><thead><tr><th><span class="">Параметр</span></th><th><span class="">Значение</span></th><th><span class="">Обоснование</span></th></tr></thead><tbody><tr><td><code>fire_confidence</code></td><td><strong><span class="">0.30</span></strong></td><td><span class="">Модель редко превышает 0.35 на fire в реальных сценах; при 0.4 fire не детектируется вовсе</span></td></tr><tr><td><code>smoke_confidence</code></td><td><strong><span class="">0.35</span></strong></td><td><span class="">Smoke уверенно детектируется от 0.40+; порог 0.35 даёт запас на ранние стадии</span></td></tr></tbody></table>

### Фильтрация hard negatives
См. [src/analytics/smoke_fire_filter.py](src/analytics/smoke_fire_filter.py). Перед генерацией события детекция проверяется по геометрическим признакам:
- `min_area_px: 2000` (~45×45 px) — отбрасывает мелкие пятна (отражения, далёкие источники света, шум).
- `max_aspect_ratio: 5.0` — отбрасывает сильно вытянутые пятна (сварка, солнечные блики, фары).

### Temporal confirmation и latency
<table><thead><tr><th><span class="">Событие</span></th><th><span class="">N/M</span></th><th><span class="">Latency (факт)</span></th><th><span class="">Цель</span></th></tr></thead><tbody><tr><td><code>smoke_detected</code></td><td><span class="">3/5</span></td><td><strong><span class="">0.34 с</span></strong></td><td><span class="">≤ 5 с ✅</span></td></tr><tr><td><code>fire_detected</code></td><td><span class="">2/3</span></td><td><strong><span class="">0.64 с</span></strong></td><td><span class="">≤ 5 с ✅</span></td></tr></tbody></table>


## 🎯 Подбор confidence threshold

Эксперимент на PPE test split (YOLO26n):

<table><thead><tr><th><span class="">conf</span></th><th><span class="">Precision</span></th><th><span class="">Recall</span></th><th><span class="">F1</span></th></tr></thead><tbody><tr><td><strong><span class="">0.30</span></strong></td><td><span class="">0.9137</span></td><td><strong><span class="">0.8937</span></strong></td><td><strong><span class="">0.9036</span></strong></td></tr><tr><td><strong><span class="">0.50</span></strong></td><td><span class="">0.9406</span></td><td><span class="">0.8631</span></td><td><span class="">0.9002</span></td></tr><tr><td><strong><span class="">0.70</span></strong></td><td><strong><span class="">0.9753</span></strong></td><td><span class="">0.7406</span></td><td><span class="">0.8419</span></td></tr></tbody></table>

**Рабочий порог:** 0.50. F1 практически максимален (разница с 0.30 — 0.3%, в пределах шума), но Precision выше — меньше ложных тревог. Порог 0.70 отклонён: Recall падает до 0.74, что недопустимо для систем безопасности (пропуск опасности критичнее ложной тревоги).



### 📦 Датасет PPE

Для валидации используется открытый датасет [jhboyo/ppe-dataset]("https://huggingface.co/datasets/jhboyo/ppe-dataset") (классы: helmet, head, vest, лицензия MIT).

### Скачивание (только тестовая часть)

```bash
pip install huggingface-hub
hf download jhboyo/ppe-dataset \
  --repo-type dataset \
  --include "test/**" \
  --local-dir ./datasets/ppe_test_only
```

После скачивания создайте `datasets/ppe_test_only/data.yaml`:

```yaml
path: /app/datasets/ppe_test_only
train: test/images # фиктивно, для совместимости с Ultralytics
val: test/images # фиктивно, для совместимости с Ultralytics
test: test/images

nc: 3
names:
    0: helmet
    1: head
    2: vest
```

Подробности: [docs/dataset_card.md](docs/dataset_card.md).

## 📊 Валидация и бенчмарк

**Валидация YOLO26n на test split** (venv или Docker):

```bash
python scripts/validate_models.py
# или
docker compose run --rm vision-app python scripts/validate_models.py
```

**Подбор порога confidence:**

```bash
python scripts/tune_confidence.py
```
**Сравнение скорости YOLO26n vs YOLOv8n:**

```bash
python scripts/benchmark_speed.py
```

**Tracking + ROI + Events (только в venv, требуется дисплей):**

```bash
python scripts/test_tracking_roi.py
```

**Контроль СИЗ (только в venv, требуется дисплей):**
```bash
python scripts/test_ppe.py
```

**Детекция дыма и огня** (только в venv):
```bash
python scripts/test_smoke_fire.py
```

**Диагностика модели fire/smoke** (для отладки порогов):
```bash
python scripts/debug_fire_model.py
```

## ⚙️ Конфигурация

Все параметры вынесены в `configs/configuration.yaml` (без hardcode):

```yaml
camera:
  id: "cam_01"
  source: "data/test_video.mp4"
  reconnect_sec: 2.0

models:
  detector_path: "models/yolov8n_openvino_model/"
  person_confidence: 0.50
  ppe_model_path: "models/ppe_yolo26n_best_openvino_model/"
  ppe_confidence: 0.50
  ppe_skip: 2                    # анализировать PPE раз в N кадров
  fire_model_path: "models/fire_detection/fire-smoke.pt"
  fire_confidence: 0.30
  smoke_confidence: 0.35

smoke_fire:
  min_area_px: 2000              # ~45×45 px — фильтр мелких ложных детекций
  max_aspect_ratio: 5.0          # фильтр вытянутых бликов/сварки


zones:
  restricted_01:
    points:
      - [0.50, 0.10]
      - [0.95, 0.10]
      - [0.95, 0.95]
      - [0.50, 0.95]

ppe:
  min_person_height: 120       # px — минимальный размер для надёжного анализа
  helmet_classes: ["helmet", "head"]
  vest_class: "vest"
  bbox_padding: 0.10

events:
  person_in_zone:
    enabled: true
    confidence: 0.50
    temporal_n: 5
    temporal_m: 8
    cooldown_sec: 10
  ppe_helmet_missing:
    enabled: true
    temporal_n: 5
    temporal_m: 8
    cooldown_sec: 30
  ppe_vest_missing:
    enabled: true
    temporal_n: 5
    temporal_m: 8
    cooldown_sec: 30
  smoke_detected:
    enabled: true
    temporal_n: 3
    temporal_m: 5
    cooldown_sec: 20
  fire_detected:
    enabled: true
    temporal_n: 2
    temporal_m: 3
    cooldown_sec: 15
```

Секреты (пароли RTSP и т.п.) **не хранятся в репозитории** — используйте переменные окружения или локальный `.env` (добавлен в `.gitignore`).

## 🗺 Дорожная карта

<table><thead><tr><th><span class="">Этап</span></th><th><span class="">Тема</span></th><th><span class="">Статус</span></th></tr></thead><tbody><tr><td><span class="">1</span></td><td><span class="">Архитектура, Git, Docker-окружение</span></td><td><span class="">✅</span></td></tr><tr><td><span class="">2</span></td><td><span class="">Видеопоток, FPS, reconnect</span></td><td><span class="">✅</span></td></tr><tr><td><span class="">3</span></td><td><span class="">Baseline-детектор (YOLOv8n, COCO)</span></td><td><span class="">✅</span></td></tr><tr><td><span class="">4</span></td><td><span class="">Датасет PPE и обучение YOLO26n</span></td><td><span class="">✅</span></td></tr><tr><td><span class="">5</span></td><td><span class="">Tracking, ROI и Event Logic</span></td><td><span class="">✅</span></td></tr><tr><td><span class="">6</span></td><td><span class="">Контроль СИЗ (логика нарушений)</span></td><td><span class="">✅</span></td></tr><tr><td><span class="">7</span></td><td><span class="">Детекция дыма и огня</span></td><td><span class="">✅</span></td></tr><tr><td><span class="">8</span></td><td><span class="">Event Engine, JSON, evidence, logging</span></td><td><span class="">🚧 В разработке</span></td></tr><tr><td><span class="">9</span></td><td><span class="">REST API, Edge AI, benchmark</span></td><td><span class="">⏳</span></td></tr><tr><td><span class="">10</span></td><td><span class="">Системные испытания и защита</span></td><td><span class="">⏳</span></td></tr></tbody></table>
