FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
# 環境変数
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 作業ディレクトリ
WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# アプリ本体コピー
COPY . .

CMD ["python", "dockertest/Arm.py"]
