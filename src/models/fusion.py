import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.methods import AttentionFusion, SelfAttentionBlock

class FusionModel(nn.Module):
    def __init__(self, input_size, num_classes, dropout_rate=0.5, fusion_method='concat', attention_dim=None):
        super(FusionModel, self).__init__()
        self.fusion_method = fusion_method

        # Initialize attention modules if needed
        if fusion_method == 'attention':
            assert attention_dim is not None, "attention_dim must be specified for attention fusion"
            self.attention_module = AttentionFusion(attention_dim)
        elif fusion_method == 'self_attention':
            assert attention_dim is not None, "attention_dim must be specified for self-attention fusion"
            self.attention_module = SelfAttentionBlock(attention_dim)
        else:
            self.attention_module = None

        # Define the rest of the model
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.fc = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(128, num_classes)

    def forward(self, quantum_features, classical_features):
        # Fuse features based on the specified method
        if self.fusion_method == 'concat':
            fused_features = torch.cat((quantum_features, classical_features), dim=1)
        elif self.fusion_method in ['attention', 'self_attention']:
            fused_features = self.attention_module(quantum_features, classical_features)
        elif self.fusion_method == 'add':
            fused_features = quantum_features + classical_features
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")

        # Apply batch normalization
        x = self.batch_norm(fused_features)

        # Linear transformation
        x = self.fc(x)

        # ReLU activation
        x = self.relu(x)

        # Dropout
        x = self.dropout(x)

        # Final output layer
        output = self.output_layer(x)
        return output

    def predict(self, quantum_features, classical_features):
        self.eval()
        with torch.no_grad():
            output = self.forward(quantum_features, classical_features)
            return F.softmax(output, dim=1)