import os
import numpy as np
from src.models.fusion import FusionMethods


def fuse_and_save(quantum_features_path, classical_features_path, output_dir, methods=None):
    os.makedirs(output_dir, exist_ok=True)

    q = np.load(quantum_features_path)
    c = np.load(classical_features_path)

    if q.shape[0] != c.shape[0]:
        raise ValueError(f"Mismatched number of samples: quantum {q.shape[0]} vs classical {c.shape[0]}")

    if methods is None:
        methods = ['concat', 'add', 'dot']

    fusion = FusionMethods()
    saved = {}

    for m in methods:
        if m == 'concat':
            out = fusion.fuse_concat(q, c)
        elif m == 'add':
            out = fusion.fuse_add(q, c)
        elif m == 'dot':
            out = fusion.fuse_dot(q, c)
        else:
            raise ValueError(f"Unknown fusion method: {m}")

        out_path = os.path.join(output_dir, f"fused_features_{m}.npy")
        np.save(out_path, out)
        saved[m] = out_path

    return saved


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fuse classical and quantum feature .npy files")
    parser.add_argument('quantum', help='Path to quantum features .npy')
    parser.add_argument('classical', help='Path to classical features .npy')
    parser.add_argument('--out', '-o', required=True, help='Output directory to save fused features')
    parser.add_argument('--methods', '-m', nargs='+', help='Fusion methods to run', default=None)
    parser.add_argument('--labels', '-l', help='Optional labels .npy to copy to output dir', default=None)

    saved = fuse_and_save(
    quantum_features_path='experiments/features/BUSUCLM/quantum/trainable/train_features.npy',
    classical_features_path='experiments/features/BUSUCLM/classical/resnet/train_features.npy',
    output_dir='experiments/features/BUSUCLM/fused/trainable_resnet',
    methods=['concat', 'add', 'dot'],
)
    print('Saved fused feature files:')
    for k, v in saved.items():
        print(f"- {k}: {v}")

if __name__ == '__main__':
    main()
