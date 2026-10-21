# Базовый образ Python (версия 3.10 подходит для большинства CV-библиотек)
FROM python:3.10-slim

# Системные зависимости для OpenCV, работы с видео и компиляции некоторых пакетов
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала копируем только requirements.txt для кэширования слоёв Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект (при разработке папки будут перекрыты volumes)
COPY . .

# Команда по умолчанию — просто проверка версии Python
CMD ["python", "--version"]
