import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class NonTrainableLayer:
    """Non-trainable quantum layer with random parameters."""
    def __init__(self, n_qubits, n_layers, rand_params):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.rand_params = rand_params
        self.dev = qml.device("lightning.qubit", wires=n_qubits)
        
        @qml.qnode(self.dev, interface='torch')
        def circuit(phi):
            for j in range(n_qubits):
                qml.RY(np.pi * phi[j], wires=j)
            qml.templates.layers.RandomLayers(rand_params, list(range(n_qubits)))
            return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]
        
        self.circuit = circuit
    
    def __call__(self, phi):
        return self.circuit(phi)


class TrainableLayer(nn.Module):
    def __init__(self, n_qubits, n_layers, patch_size, mode="2d", stride=None, use_parallel=True):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.patch_size = patch_size
        self.mode = mode
        self.stride = stride if stride is not None else patch_size
        self.use_parallel = use_parallel

        self.dev = qml.device("lightning.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface='torch')
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.templates.RandomLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.qnode = qml.QNode(circuit, self.dev, interface="torch")
        self.qlayer = qml.qnn.TorchLayer(self.qnode, weight_shapes)

        print(f"Quantum layer has {sum(p.numel() for p in self.qlayer.parameters())} trainable parameters")
        print(f"Requires grad: {all(p.requires_grad for p in self.qlayer.parameters())}")

    def calculate_output_shape(self, h, w):
        """Calculate the output shape after quantum convolution."""
        p = self.patch_size
        s = self.stride
        
        if self.mode == "2d":
            pad_h = (p - (h % p)) % p if (h % p) != 0 else 0
            pad_w = (p - (w % p)) % p if (w % p) != 0 else 0
            
            h_padded = h + pad_h
            w_padded = w + pad_w
            
            out_h = (h_padded - p) // s + 1
            out_w = (w_padded - p) // s + 1
            
            total_features = self.n_qubits * out_h * out_w
            return (out_h, out_w), total_features
            
        else:  # sequential or matmul
            pad_w = (p - (w % p)) % p
            pad_h = (p - (h % p)) % p
            h_after_h_pad = h + pad_h
            w_after_h_pad = w + pad_w
            
            out_h_horiz = h_after_h_pad
            out_w_horiz = w_after_h_pad // p
            
            h_for_vert = out_h_horiz
            pad_h_vert = (p - (h_for_vert % p)) % p
            h_after_v_pad = h_for_vert + pad_h_vert
            
            out_h_final = h_after_v_pad // p
            out_w_final = out_w_horiz
            
            total_features = self.n_qubits * out_h_final * out_w_final
            return (out_h_final, out_w_final), total_features

    def _prepare_patches(self, data, dim_size):
        """Prepare patches from input data."""
        B, seq_len = data.shape
        p = self.patch_size
 
        num_patches = seq_len // p
        patches = data[:, :num_patches * p].view(B, num_patches, p)
        
        patches = patches.reshape(-1, p)
        
        if p > self.n_qubits:
            patches = patches[:, :self.n_qubits]
        elif p < self.n_qubits:
            pad_size = self.n_qubits - p
            patches = F.pad(patches, (0, pad_size))
        
        return patches, num_patches

    def quanv2d(self, x):
        """Apply 2D quantum convolution."""
        B, C, H, W = x.shape
        p = self.patch_size
        s = self.stride

        pad_h = (p - (H % p)) % p if (H % p) != 0 else 0
        pad_w = (p - (W % p)) % p if (W % p) != 0 else 0
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
            B, C, H, W = x.shape
        
        out_h = (H - p) // s + 1
        out_w = (W - p) // s + 1

        patches = x.unfold(2, p, s).unfold(3, p, s)  
        
        patches = patches.squeeze(1) 
        patches = patches.reshape(B, out_h, out_w, p * p) 
        
        patches_flat = patches.reshape(B * out_h * out_w, p * p)
        
        patch_dim = p * p
        if patch_dim > self.n_qubits:
            patches_flat = patches_flat[:, :self.n_qubits]
        elif patch_dim < self.n_qubits:
            pad_size = self.n_qubits - patch_dim
            patches_flat = F.pad(patches_flat, (0, pad_size), value=0.0)
        
        q_out = self.qlayer(patches_flat)

        # lightning.qubit always simulates on CPU: with trainable weights PennyLane
        # moves the result back to the input's device, but when the weights are frozen
        # (requires_grad=False) there is no autograd path and the tensor stays on CPU.
        # Realign it, the same way QuanvolutionMixin does for the fixed circuit.
        q_out = q_out.to(patches_flat.device)

        q_out = q_out.view(B, out_h, out_w, self.n_qubits)
        q_out = q_out.permute(0, 3, 1, 2)  
        return q_out

    def forward(self, x):
        """Forward pass through quantum layer."""
        if self.mode == "2d":
            return self.quanv2d(x)
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use '2d'.")
