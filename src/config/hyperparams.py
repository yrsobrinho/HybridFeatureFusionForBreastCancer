import json
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class HyperparameterConfig:
    """Centralized hyperparameter management"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            base_path = os.environ.get('HYBRID_EMBEDDINGS_ROOT', 
                                     '/home/eflammere/HybridEmbeddingsForSaltDome')
            config_path = os.path.join(base_path, "salt_src", "config", "hyperparams.json")
        
        self.config_path = config_path
        self._load_config()
    
    def _load_config(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            self.common = config["common"]
            self.quantum = config["quantum"] 
            self.classical = config["classical"]
            self.fusion = config["fusion"]
            
            print(f"Hyperparameters loaded from: {self.config_path}")
            
        except FileNotFoundError:
            print(f"Config file not found: {self.config_path}")
            print(" Using default hyperparameters...")
            self._set_defaults()
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in config file: {e}")
            print(" Using default hyperparameters...")
            self._set_defaults()
    
    def _set_defaults(self):
        """Set default hyperparameters if config file fails"""
        self.common = {
            "epochs_phase1": 20,
            "epochs_phase2": 10,
            "batch_size": 256,
            "num_classes": 2,
            "attention_dim": 128,
            "temperature": 1.0,
            "alpha": 0.2,
            "alpha_max": 0.9,
            "alpha_min": 0.1,
            "beta": 0.1,
            "fusion_method": "concat"
        }
        self.quantum = {
            "n_qubits": 4,
            "n_layers": 1,
            "learning_rate": 1e-4
        }
        self.classical = {
            "learning_rate": 1e-4
        }
        self.fusion = {
            "learning_rate": 1e-3
        }
    
    def get_all_params(self):
        """Get all parameters as a tuple (for backward compatibility)"""
        return self.common, self.quantum, self.classical, self.fusion