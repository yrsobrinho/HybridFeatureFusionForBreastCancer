import os
import numpy as np
from src.utils.fusion import FusionMethods


def fuse_and_save(quantum_features_path, classical_features_path, output_dir, methods=None):
    """Load quantum and classical feature .npy files, apply fusion methods and save results.

    Args:
        quantum_features_path: path to numpy file with quantum features (N x Dq)
        classical_features_path: path to numpy file with classical features (N x Dc)
        output_dir: directory where fused features will be saved
        methods: list of method names to run; defaults to ['concat','add','dot','attention','self_attention']
        labels_path: optional path to labels .npy; if provided, labels will be copied to output_dir
    Returns:
        dict mapping method name to saved file path
    """
    os.makedirs(output_dir, exist_ok=True)

    q = np.load(quantum_features_path)
    c = np.load(classical_features_path)

    if q.shape[0] != c.shape[0]:
        raise ValueError(f"Mismatched number of samples: quantum {q.shape[0]} vs classical {c.shape[0]}")

    if methods is None:
        methods = ['concat', 'add', 'dot', 'attention', 'self_attention']

    fusion = FusionMethods()
    saved = {}

    for m in methods:
        if m == 'concat':
            out = fusion.fuse_concat(q, c)
        elif m == 'add':
            out = fusion.fuse_add(q, c)
        elif m == 'dot':
            out = fusion.fuse_dot(q, c)
        elif m == 'attention':
            out = fusion.fuse_attention(q, c)
        elif m == 'self_attention':
            out = fusion.fuse_self_attention(q, c)
        else:
            raise ValueError(f"Unknown fusion method: {m}")

        out_path = os.path.join(output_dir, f"fused_features_{m}.npy")
        np.save(out_path, out)
        saved[m] = out_path

    return saved


def main():
    """Small CLI to fuse two feature files. Edit paths below or import fuse_and_save."""
    import argparse

    parser = argparse.ArgumentParser(description="Fuse classical and quantum feature .npy files")
    parser.add_argument('quantum', help='Path to quantum features .npy')
    parser.add_argument('classical', help='Path to classical features .npy')
    parser.add_argument('--out', '-o', required=True, help='Output directory to save fused features')
    parser.add_argument('--methods', '-m', nargs='+', help='Fusion methods to run', default=None)
    parser.add_argument('--labels', '-l', help='Optional labels .npy to copy to output dir', default=None)

    saved = fuse_and_save(
    quantum_features_path='/home/eflammere/HybridEmbeddingsForBreastCancerDetection/experiments/features/quantum/train_features.npy',
    classical_features_path='/home/eflammere/HybridEmbeddingsForBreastCancerDetection/experiments/features/classical/train_features.npy',
    output_dir='/home/eflammere/HybridEmbeddingsForBreastCancerDetection/experiments/features/fused',
    methods=['concat', 'add', 'dot', 'attention', 'self_attention'],
)
    print('Saved fused feature files:')
    for k, v in saved.items():
        print(f"- {k}: {v}")

if __name__ == '__main__':
    main()
