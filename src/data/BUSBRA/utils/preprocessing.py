import cv2
import numpy as np
import pandas as pd
import os
from ast import literal_eval
from sklearn.model_selection import train_test_split

min_size = 64  
target_size = 224  
images_dir = "src/data/BUSBRA/original_png"  
output_base_dir = "src/data/BUSBRA/png"  
csv_path = "src/data/BUSBRA/csv/bus_data.csv"

train_dir = os.path.join(output_base_dir, "train")
val_dir = os.path.join(output_base_dir, "val")
test_dir = os.path.join(output_base_dir, "test")

for folder in [train_dir, val_dir, test_dir]:
    os.makedirs(os.path.join(folder, "0"), exist_ok=True) 
    os.makedirs(os.path.join(folder, "1"), exist_ok=True) 

# CLAHE com parâmetros ajustados para melhor contraste
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# Kernel para sharpening (realçar bordas)
sharpening_kernel = np.array([[-1, -1, -1], 
                               [-1,  9, -1], 
                               [-1, -1, -1]])

def process_csv(csv_path):
    df = pd.read_csv(csv_path)

    df["label"] = df["Pathology"].str.lower().apply(lambda x: "0" if x == "benign" else "1")

    # Split: 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

    for dataset, data_type in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        for _, row in dataset.iterrows():
            image_name = row["ID"].strip() + ".png" 
            image_path = os.path.join(images_dir, image_name)

            classification = row["label"]
            output_path = os.path.join(output_base_dir, data_type, classification, image_name)

            if not os.path.exists(image_path):
                print(f"Imagem não encontrada: {image_path}")
                continue
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Erro ao carregar imagem: {image_path}")
                continue

            try:
                x_min, y_min, x_max, y_max = literal_eval(row["BBOX"])
            except Exception as e:
                print(f"Erro ao processar BBOX para {image_name}: {e}")
                continue

            x_min, x_max = max(0, x_min), min(image.shape[1], x_max)
            y_min, y_max = max(0, y_min), min(image.shape[0], y_max)

            if (x_max - x_min) < min_size:
                diff = (min_size - (x_max - x_min)) // 2
                x_min = max(0, x_min - diff)
                x_max = min(image.shape[1], x_max + diff)

            if (y_max - y_min) < min_size:
                diff = (min_size - (y_max - y_min)) // 2
                y_min = max(0, y_min - diff)
                y_max = min(image.shape[0], y_max + diff)

            cropped_image = image[y_min:y_max, x_min:x_max]

            # 1. Redução de ruído (preserva bordas)
            denoised = cv2.bilateralFilter(cropped_image, d=5, sigmaColor=50, sigmaSpace=50)

            # 2. Aplicar CLAHE no canal L (LAB color space)
            lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = clahe.apply(l)  
            lab = cv2.merge((l, a, b))
            enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            # 3. Sharpening para realçar bordas
            sharpened = cv2.filter2D(enhanced_image, -1, sharpening_kernel)

            # 4. Resize direto para 224x224
            resized_image = cv2.resize(sharpened, (target_size, target_size), interpolation=cv2.INTER_AREA)

            cv2.imwrite(output_path, resized_image)
            print(f"Saved: {output_path}")

process_csv(csv_path)
