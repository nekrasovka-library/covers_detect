FROM python:3.9-slim
WORKDIR /

ARG TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN mkdir -p /workdir

RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
COPY requirements.txt .
RUN pip install --progress-bar=off -U --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]