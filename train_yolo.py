from ultralytics import YOLO
import torch
import os
import subprocess

# ==========================
# CONFIGURAÇÕES
# ==========================
DATA_YAML = "dataset_yolo/data.yaml"
MODEL_BASE = "yolov8s.pt"   # recomendo 's' para documentos
EPOCHS = 11
IMG_SIZE = 640
BATCH_SIZE = 16
WORKERS = 8
PROJECT = "runs_yolo"
NAME = "nivel1_documentos"
MODEL_BUCKET = "amhp-models"

# CHECK GPU
print("CUDA disponível:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# LOAD MODEL
model = YOLO(MODEL_BASE)

# TREINAMENTO
model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    workers=WORKERS,
    device=0,
    project=PROJECT,
    name=NAME,
    pretrained=True,
    optimizer="AdamW",
    lr0=1e-3,
    patience=10,
    cos_lr=True,
    verbose=True
)

# EXPORT FINAL
best_model_path = os.path.join(PROJECT, NAME, "weights", "best.pt")
print(f"\n✅ Treinamento concluído")
print(f"📦 Melhor modelo salvo em: {best_model_path}")

print("☁️ Enviando modelo para o S3...")
subprocess.run(
    ["aws", "s3", "cp", best_model_path, f"s3://{MODEL_BUCKET}/"],
    check=True
)
