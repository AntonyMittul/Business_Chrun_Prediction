# Churn Prediction API — self-contained serving image.
# The data pipeline + training run AT BUILD TIME (deterministic, seed 42), so the
# image never depends on artifacts existing on the host. Training needs a couple
# of extra libraries not in the serving set; they're installed only for the build
# stage of the layer.
FROM python:3.12-slim

WORKDIR /app

# libgomp1: OpenMP runtime required by LightGBM wheels
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt \
    && pip install --no-cache-dir "mlflow>=2.12"  # train.py logs here at build

COPY config/ config/
COPY src/ src/
COPY data/raw/ data/raw/

# Build the artifacts inside the image: clean -> features -> train
RUN python -m src.data.clean_data \
    && python -m src.features.build_features \
    && python -m src.models.train

EXPOSE 8080
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
