"""Case-wise (patient-wise) partitioning for BUS-UCLM.

Marked frames in BUS-UCLM largely duplicate an unmarked frame of the same case,
so a random split would put the same lesion in both train and test and inflate
the reported score. Keying the split on the 4-letter case code prevents that.

This module holds no heavy dependencies on purpose: both the dataloader and the
dataset preprocessing script import it, so the two produce the same partition
instead of each rolling its own draw.
"""

import os
import csv
import random

# Fixed by the official BUS-UCLM partition script.
BUSUCLM_TEST_CASES = ('COPE', 'ANFO', 'ELCO', 'CRCI', 'FLKA')
BUSUCLM_LABELS = {'benign': 0, 'malignant': 1}


def busuclm_case_id(image_name):
    """BUS-UCLM filenames are <4-letter case code>_<index>.png."""
    return os.path.basename(image_name)[:4]


def busuclm_pick_val_cases(cases, sizes, malignant_counts, val_frac, seed):
    """Draw whole cases for validation until val_frac of the images is reached.

    A malignant-bearing case is forced in first: the pool is benign-heavy enough
    that a plain draw can hand back a single-class validation set.
    """
    if val_frac <= 0:
        return []

    order = sorted(cases)
    random.Random(seed).shuffle(order)

    malignant_cases = [c for c in order if malignant_counts[c] > 0]
    if malignant_cases:
        order.insert(0, order.pop(order.index(malignant_cases[0])))

    target = val_frac * sum(sizes.values())
    chosen, taken = [], 0

    for case in order:
        if taken >= target:
            break
        chosen.append(case)
        taken += sizes[case]

    return sorted(chosen)


def busuclm_read_info(data_root):
    """Yield (image_path, label, case) for every classifiable BUS-UCLM row.

    Reads INFO_clean.csv when present, otherwise the raw INFO.csv; either way
    Normal rows are dropped, since they carry no lesion to classify.
    """
    data_root = str(data_root)
    images_dir = os.path.join(data_root, 'images')

    csv_path = os.path.join(data_root, 'INFO_clean.csv')
    if not os.path.isfile(csv_path):
        csv_path = os.path.join(data_root, 'INFO.csv')

    if not os.path.isfile(csv_path) or not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"Expected images/ and INFO.csv under {data_root} - "
            "run src/data/BUS-UCLM/utils/preprocessing.py first."
        )

    samples, missing = [], []

    # The release ships INFO.csv semicolon-separated and with a UTF-8 BOM.
    with open(csv_path, newline='', encoding='utf-8-sig') as handle:
        for row in csv.DictReader(handle, delimiter=';'):
            image_name = row['Image'].strip()
            label = BUSUCLM_LABELS.get(row['Label'].strip().lower())

            if label is None:
                continue

            image_path = os.path.join(images_dir, image_name)
            if not os.path.isfile(image_path):
                missing.append(image_name)
                continue

            samples.append((image_path, label, busuclm_case_id(image_name)))

    if missing:
        print(f"Warning: {len(missing)} images listed in {os.path.basename(csv_path)} "
              f"are absent from {images_dir}")

    return samples


def busuclm_case_split(data_root, test_cases=BUSUCLM_TEST_CASES, val_frac=0.15, seed=42):
    """Partition BUS-UCLM by case. Returns {split: [(image_path, label), ...]}."""
    samples = busuclm_read_info(data_root)
    test_cases = set(test_cases)

    sizes, malignant_counts = {}, {}
    for _, label, case in samples:
        if case in test_cases:
            continue
        sizes[case] = sizes.get(case, 0) + 1
        malignant_counts[case] = malignant_counts.get(case, 0) + label

    val_cases = set(busuclm_pick_val_cases(sizes, sizes, malignant_counts, val_frac, seed))

    split = {'train': [], 'val': [], 'test': []}
    for image_path, label, case in samples:
        if case in test_cases:
            name = 'test'
        elif case in val_cases:
            name = 'val'
        else:
            name = 'train'
        split[name].append((image_path, label))

    return split
