import os
import shutil
import subprocess
from pathlib import Path
from tqdm import tqdm
import random

# ==========================
# CONFIGURAÇÕES
# ==========================
DATA_BUCKET = "amhp-teste-ml/cid-yolo-dataset"
AWS_REGION = "us-east-1"

DATA_DIR = Path("data_s3")           # onde o aws s3 sync vai baixar
LOCAL_DATASET = Path("dataset_yolo") # dataset final YOLO

IMG_EXT = (".jpg", ".jpeg", ".png")

CLASSES = {
    "ASSINATURAS_DESCRICAO_HSL": 0,
    "ASSINATURAS_GDF": 1,
    "AUTORIZACAO_CONSULTA": 2,
    "AUTORIZACAO_HONORARIO": 3,
    "AUTORIZACAO_SADT": 4,
    "AUTORIZACAO_TEXTO": 5,
    "BOLETIM": 6,
    "DESCRICAO_CIRURGICA": 7,
    "DOCUMENTO_CARTEIRINHA": 8,
    "DOCUMENTO_CERTIDAO": 9,
    "ELEGIBILIDADE": 10, 
    "FICHA_INTERNACAO": 11,
    "GUIA_CBMDF": 12,
    "GUIA_CONSULTA": 13,
    "GUIA_HONORARIO": 14,
    "GUIA_OUTRAS_DESPESAS": 15,
    "GUIA_PERIODICO": 16,
    "GUIA_RESUMO": 17,
    "GUIA_SADT": 18,
    "GUIA_SISTEMA_CONSULTA": 19,
    "GUIA_SISTEMA_SADT": 20,
    "HMED": 21,
    "LACRE": 22,
    "PAGINA_BRANCO": 23,
    "PEDIDO_MEDICO": 24,
    "RELATORIO_MEDICO": 25,
    "SOLICITACAO_INTERNACAO": 26
}

TRAIN_RATIO = 0.8  # 80% treino / 20% validação

# CRIA DIRETÓRIOS
DATA_DIR.mkdir(parents=True, exist_ok=True)

for split in ["train", "val"]:
    (LOCAL_DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
    (LOCAL_DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

# DOWNLOAD DO DATASET (UMA VEZ)
print("⬇️ Sincronizando dataset do S3 (download incremental)...")
subprocess.run(
    [
        "aws", "s3", "sync",
        f"s3://{DATA_BUCKET}",
        str(DATA_DIR),
        "--region", AWS_REGION,
        "--only-show-errors"
    ],
    check=True
)

print("✅ Download concluído")

# PROCESSAMENTO LOCAL
for classe_nome, class_id in CLASSES.items():
    pasta_classe = DATA_DIR / classe_nome

    if not pasta_classe.exists():
        print(f"⚠️ Pasta não encontrada: {pasta_classe}")
        continue

    imagens = [
        p for p in pasta_classe.iterdir()
        if p.suffix.lower() in IMG_EXT
    ]

    if not imagens:
        print(f"⚠️ Nenhuma imagem em {classe_nome}")
        continue

    random.shuffle(imagens)
    split_idx = int(len(imagens) * TRAIN_RATIO)

    print(f"\n📂 Processando {classe_nome} ({len(imagens)} imagens)")

    for idx, img_path in enumerate(tqdm(imagens)):
        split = "train" if idx < split_idx else "val"

        new_name = f"{classe_nome}_{img_path.name}"

        img_dest = LOCAL_DATASET / "images" / split / new_name
        label_dest = LOCAL_DATASET / "labels" / split / f"{Path(new_name).stem}.txt"

        # shutil.copy(img_path, img_dest)

        try:
            os.link(img_path, img_dest)   # hard link (zero espaço)
        except OSError:
            shutil.copy(img_path, img_dest)  # fallback se não suportar

        # Label YOLO – página inteira
        with open(label_dest, "w") as f:
            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

print("\n✅ Dataset YOLO preparado com sucesso (via aws s3 sync)")
