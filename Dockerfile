FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN pip install --upgrade pip \
    && pip install torch torch-geometric scipy

COPY GNN.py .

CMD ["python", "GNN.py"]
