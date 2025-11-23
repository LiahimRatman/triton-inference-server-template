# Базовый образ Triton с Python backend
FROM --platform=linux/amd64 nvcr.io/nvidia/tritonserver:23.10-py3

# Обновляем систему (минимально) и ставим git (иногда нужен HF/torch)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Рабочая директория
WORKDIR /opt/triton

# Кладём файлы poetry
COPY pyproject.toml ./
# Если у тебя есть poetry.lock — раскомментируй следующую строку:
# COPY poetry.lock ./

# Ставим Poetry и зависимости (без virtualenv внутри контейнера)
RUN pip install pip poetry && poetry config virtualenvs.create false
RUN poetry install --no-root --all-extras

RUN pip uninstall -y torch torchvision torchaudio || true \
 && pip install \
      torch==2.6.0 torchvision==0.21.0 \
      --index-url https://download.pytorch.org/whl/cu118

# Копируем repo моделей Triton
COPY models /models

# Порты Triton
EXPOSE 8000 8001 8002

# Стартуем Triton
CMD ["tritonserver", "--model-repository=/models", "--log-verbose=1"]
