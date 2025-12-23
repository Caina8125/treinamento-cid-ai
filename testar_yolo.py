from ultralytics import YOLO
from pathlib import Path
import torch

# CONFIGURAÇÕES
MODEL_PATH = "best.pt"  # caminho local do modelo
SOURCE = R"C:\treinamento-cid-ai\teste_censo\solict_hono_geap2.png"  # pode ser: imagem.jpg OU pasta
CONF_THRESHOLD = 0.25

# # CHECK GPU
# print("CUDA disponível:", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("GPU:", torch.cuda.get_device_name(0))

device = "cpu"

# LOAD MODEL
model = YOLO(MODEL_PATH)

# INFERÊNCIA
results = model(
    source=SOURCE,
    conf=CONF_THRESHOLD,
    imgsz=640,
    device=device
)

# RESULTADOS
for r in results:
    if r.boxes is None or len(r.boxes) == 0:
        print(f"❌ Nenhuma detecção em {r.path}")
        continue

    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]

        print(f"📄 Arquivo: {Path(r.path).name}")
        print(f"   ➜ Classe: {class_name}")
        print(f"   ➜ Confiança: {conf:.3f}")
        print("-" * 40)
