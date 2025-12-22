import os
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import requests

# ===============================
# CONFIGURAÇÕES
# ===============================
DATA_BUCKET = "amhp-cid-dataset"
MODEL_BUCKET = "amhp-models"
AWS_REGION = "sa-east-1"

BASE_DIR = "/home/ec2-user"
DATA_DIR = f"{BASE_DIR}/project/data"
MODEL_DIR = f"{BASE_DIR}/models"
LOG_DIR = f"{BASE_DIR}/logs"

BATCH_SIZE = 32
EPOCHS = 11
LR = 1e-4

MODEL_PTH = f"{MODEL_DIR}/AMHP_CID_AI.pth"
# MODEL_TS  = f"{MODEL_DIR}/AMHP_CID_AI.ts"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Device: {device}")

# 1️⃣ DOWNLOAD DO DATASET (SE NECESSÁRIO)
print("⬇️ Sincronizando dataset do S3 (download incremental)...")
subprocess.run(
    [
        "aws", "s3", "sync",
        f"s3://{DATA_BUCKET}",
        DATA_DIR,
        "--region", AWS_REGION,
        "--only-show-errors"
    ],
    check=True
)


# TRANSFORMS (IGUAL AO COLAB)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# DATASET / SPLIT
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=8, pin_memory=True, persistent_workers=True
)

val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=8, pin_memory=True, persistent_workers=True
)

print(f"📊 Total imagens: {len(dataset)} | Classes: {dataset.classes}")

# MODELO
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# VALIDAÇÃO
def validate():
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            _, pred = out.max(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100 * correct / total

# TREINAMENTO
print("🚀 Iniciando treinamento")
for epoch in range(EPOCHS):
    model.train()
    correct = total = loss_sum = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for x, y in loop:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        _, pred = out.max(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    train_acc = 100 * correct / total
    val_acc = validate()

    print(
        f"Epoch {epoch+1} | "
        f"Loss {loss_sum/len(train_loader):.4f} | "
        f"Train {train_acc:.2f}% | Val {val_acc:.2f}%"
    )

# SALVAR MODELOS
torch.save(model.state_dict(), MODEL_PTH)
print("💾 Modelo salvo localmente")

# UPLOAD PARA S3
print("☁️ Enviando modelo para o S3...")
subprocess.run(
    ["aws", "s3", "cp", MODEL_PTH, f"s3://{MODEL_BUCKET}/"],
    check=True
)


# DESLIGAR A INSTÂNCIA
print("🛑 Desligando instância EC2...")
instance_id = requests.get(
    "http://169.254.169.254/latest/meta-data/instance-id", timeout=2
).text

subprocess.run(
    ["aws", "ec2", "stop-instances", "--instance-ids", "i-04756ed93f0692f40", "--region", AWS_REGION],
    check=True
)

print("✅ Job finalizado com sucesso")
