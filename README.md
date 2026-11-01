# Модуль промышленного зрения

Разрабатываемый модуль предназначен для автоматизированного анализа видеопотока, получаемого от камеры мобильной роботизированной платформы, выполняющей инспекционный обход промышленной территории. Основная обработка изображений производится локально на бортовом вычислительном устройстве, что позволяет снизить зависимость от внешнего канала связи и уменьшить задержку реакции на события.

Модуль должен принимать видеопоток, извлекать кадры, выполнять предварительную обработку, запускать модели компьютерного зрения, отслеживать обнаруженные объекты, анализировать контролируемые области изображения, формировать события, сохранять подтверждающие изображения, вести журнал и предоставлять результаты внешнему программному интерфейсу.

## 📋 Назначение

Модуль предназначен для:

- Получения видеопотока (файл, USB-камера, RTSP-поток).
- Детекции людей и транспорта (baseline-модель).
- Контроля средств индивидуальной защиты (СИЗ): каска и жилет.
- Формирования событий безопасности на основе ROI.
- Предоставления результатов через REST API.

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
│   ├── test_capture.py
│   ├── test_detector.py
│   ├── test_tracking_roi.py       # Демонстрация Tracking + ROI + Event Engine
│   ├── tune_confidence.py         # Подбор порога (3 прогона val)
│   └── validate_models.py
├── src/                           # Исходный код
│   ├── analytics/                 # Аналитика
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

## 🚀 Установка и запуск (Docker)

### Требования

- Docker Desktop (Windows/macOS) или Docker Engine (Linux).
- Docker Compose v2.
- ~2 ГБ свободного места для образа и моделей.

### 1. Сборка образа

```bash
docker compose build
```

### 2. Подготовка данных

Положите тестовое видео в `data/test_video.mp4` (или используйте своё).

### 3. Запуск проверок

Проверка чтения видео и FPS:

```bash
docker compose run --rm vision-app python scripts/test_capture.py
```

Проверка baseline-детектора (YOLOv8n, COCO):

```bash
docker compose run --rm vision-app python scripts/test_detector.py
```

## 🧠 Модели

Проект использует две модели для разных задач:

<table><thead><tr><th><span class="">Модель</span></th><th><span class="">Задача</span></th><th><span class="">Классы</span></th><th><span class="">Статус</span></th></tr></thead><tbody><tr><td><strong><span class="">YOLOv8n (COCO)</span></strong></td><td><span class="">Детекция людей и транспорта</span></td><td><code>person</code><span class="">, </span><code>car</code><span class="">, </span><code>truck</code><span class="">, </span><code>bus</code><span class="">, </span><code>motorcycle</code></td><td><span class="">✅ Интегрирована</span></td></tr><tr><td><strong><span class="">YOLO26n (fine-tuned)</span></strong></td><td><span class="">Контроль СИЗ</span></td><td><code>helmet</code><span class="">, </span><code>head</code><span class="">, </span><code>vest</code></td><td><span class="">✅ Обучена и валидирована</span></td></tr></tbody></table>

### Метрики YOLO26n (PPE)

<table><thead><tr><th><span class="">Метрика</span></th><th><span class="">Значение</span></th></tr></thead><tbody><tr><td><span class="">Precision</span></td><td><strong><span class="">0.9137</span></strong></td></tr><tr><td><span class="">Recall</span></td><td><strong><span class="">0.8937</span></strong></td></tr><tr><td><span class="">mAP@0.5</span></td><td><strong><span class="">0.9492</span></strong></td></tr><tr><td><span class="">mAP@0.5:0.95</span></td><td><strong><span class="">0.6867</span></strong></td></tr></tbody></table>

Подробности: [`docs/model_card.md`](docs/model_card.md).

### Где взять веса моделей

- `models/yolov8n.pt` — скачивается автоматически при первом запуске `test_detector.py`.
- `models/ppe_yolo26n_best.pt` — обученная в Google Colab модель. Скачайте файл и положите в `models/`.

## 🎯 Tracking (Этап 5)

Для присвоения стабильных Track ID используется ByteTrack с кастомной конфигурацией configs/bytetrack_custom.yaml:

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

## ⚙️ Конфигурация

Все параметры вынесены в `configs/configuration.yaml` (без hardcode):

```yaml
camera:
    id: "cam_01"
    source: "data/test_video.mp4"
    reconnect_sec: 2.0

models:
    detector_path: "models/yolov8n.pt"
    person_confidence: 0.50
    fire_confidence: 0.65

zones:
    restricted_01:
        points:
              - [0.50, 0.10]
              - [0.95, 0.10]
              - [0.95, 0.95]
              - [0.50, 0.95]
        
events:
    person_in_zone:
    enabled: true
    confidence: 0.50
    temporal_n: 5
    temporal_m: 8
    cooldown_sec: 10
```

Секреты (пароли RTSP и т.п.) **не хранятся в репозитории** — используйте переменные окружения или локальный `.env` (добавлен в `.gitignore`).

## 🗺 Дорожная карта

<table><thead><tr><th><span class="">Этап</span></th><th><span class="">Тема</span></th><th><span class="">Статус</span></th></tr></thead><tbody><tr><td><span class="">1</span></td><td><span class="">Архитектура, Git, Docker-окружение</span></td><td><span class="">✅</span></td></tr><tr><td><span class="">2</span></td><td><span class="">Видеопоток, FPS, reconnect</span></td><td><span class="">✅</span></td></tr><tr><td><span class="">3</span></td><td><span class="">Baseline-детектор (YOLOv8n, COCO)</span></td><td><span class="">✅</span></td></tr><tr><td><span class="">4</span></td><td><span class="">Датасет PPE и обучение YOLO26n</span></td><td><span class="">✅</span></td></tr><tr><td><span class="">5</span></td><td><span class="">Tracking, ROI и Event Logic</span></td><td><span class="">✅</span></td></tr><tr><td><span class="">6</span></td><td><span class="">Контроль СИЗ (логика нарушений)</span></td><td><span class="">🚧 В разработке</span></td></tr><tr><td><span class="">7</span></td><td><span class="">Детекция дыма и огня</span></td><td><span class="">⏳</span></td></tr><tr><td><span class="">8</span></td><td><span class="">Event Engine, JSON, evidence, logging</span></td><td><span class="">⏳</span></td></tr><tr><td><span class="">9</span></td><td><span class="">REST API, Edge AI, benchmark</span></td><td><span class="">⏳</span></td></tr><tr><td><span class="">10</span></td><td><span class="">Системные испытания и защита</span></td><td><span class="">⏳</span></td></tr></tbody></table>
