import os
import numpy as np
import pandas as pd

def check_shapes(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if filename.endswith(".npy"):
            data = np.load(file_path)
            print(f"{filename}: shape {data.shape}")
        
        elif filename.endswith(".csv"):
            df = pd.read_csv(file_path)
            print(f"{filename}: shape {df.shape}")

directory = "/home/eflammere/HybridEmbeddingsForBreastCancerDetection/embeddings/classical"
check_shapes(directory)