import os
import numpy as np
import torch
from tqdm import tqdm

class FeatureExtractor:
    def __init__(self, model, device=None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.to(self.device)

    def extract_features_quantum(self, model, dataloader, dataset_name, output_dir, device):
        os.makedirs(output_dir, exist_ok=True)

        features_list, labels_list = [], []

        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc=f"Extracting Quantum Features ({dataset_name})"):
                images = images.to(device)

                # Extract features using the quantum model
                features = model(images)
                features_list.append(features.cpu().numpy())
                labels_list.append(labels.cpu().numpy())

        # Concatenate all features and labels
        features_array = np.concatenate(features_list, axis=0)
        labels_array = np.concatenate(labels_list, axis=0)

        # Save features and labels as .npy files
        np.save(os.path.join(output_dir, f"{dataset_name}_features.npy"), features_array)
        np.save(os.path.join(output_dir, f"{dataset_name}_labels.npy"), labels_array)

        print(f"{dataset_name} quantum features saved in {output_dir}.")

    def extract_features_classical(self, model, dataloader, dataset_name, output_dir, device):
        os.makedirs(output_dir, exist_ok=True)

        features_list, labels_list = [], []

        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc=f"Extracting Features ({dataset_name})"):
                images = images.to(device)

                # Extract features using the classical model
                features = model(images)
                features = features.view(features.size(0), -1)  # Flatten the features

                features_list.append(features.cpu().numpy())
                labels_list.append(labels.cpu().numpy())

        # Concatenate all features and labels
        features_array = np.concatenate(features_list, axis=0)
        labels_array = np.concatenate(labels_list, axis=0)

        # Save features and labels as .npy files
        np.save(os.path.join(output_dir, f"{dataset_name}_features.npy"), features_array)
        np.save(os.path.join(output_dir, f"{dataset_name}_labels.npy"), labels_array)

        print(f"{dataset_name} classical features saved in {output_dir}.")