"""Prepare the BUSUCLM dataset.

Two stages, run in order:

  clean  The raw release ships 683 PNGs. --rule picks what to drop from
         `images/`; the default drops Label=Normal (no lesion to classify) and
         keeps Marks=Yes, whose calipers/annotations are burned into the frame.
         Marked frames largely duplicate an unmarked frame of the same case,
         so they must never be split randomly - see the split stage below.
         Discarded files are moved to a sibling `removed/` folder rather than
         erased, and the stage reconciles in both directions, so loosening the
         rule and re-running restores what an earlier run took out. Nothing is
         touched unless --apply is passed.

  split  Partitions the surviving images into png/{train,val,test}/{0,1}, with
         0 = Benign and 1 = Malignant, matching what BUSUCLMDataset expects.
         The split is case-wise (patient-wise): every image whose filename
         starts with the same 4-letter case code lands in the same partition,
         so no patient leaks across partitions. The test cases are the ones
         fixed by the official BUS-UCLM partition script; validation cases are
         held out from the remaining training cases with a seeded draw.
"""

import os
import sys
import shutil
import argparse
import pandas as pd

# This script lives under src/data/BUS-UCLM/, which is not an importable package
# path (the hyphen is not a valid identifier), so reach the repo root explicitly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

from src.utils.case_split import BUSUCLM_TEST_CASES, busuclm_pick_val_cases

data_root = "src/data/BUS-UCLM"
csv_path = os.path.join(data_root, "INFO.csv")
images_dir = os.path.join(data_root, "images")
removed_dir = os.path.join(data_root, "removed")
clean_csv_path = os.path.join(data_root, "INFO_clean.csv")
split_dir = os.path.join(data_root, "png")

test_cases = list(BUSUCLM_TEST_CASES)

label_map = {'benign': '0', 'malignant': '1'}


def load_info(csv_file):
    # The release ships INFO.csv semicolon-separated and with a UTF-8 BOM.
    data = pd.read_csv(csv_file, sep=';', encoding='utf-8-sig')

    for column in ['Image', 'Label', 'Marks']:
        data[column] = data[column].astype(str).str.strip()

    return data


def flag_discards(data, rule):
    marked = data['Marks'].str.lower() == 'yes'
    normal = data['Label'].str.lower() == 'normal'

    discard = {
        'any': marked | normal,
        'all': marked & normal,
        'normal': normal,
        'marks': marked,
    }[rule]

    print(f"Total images:     {len(data)}")
    print(f"  Marks = Yes:    {marked.sum()}")
    print(f"  Label = Normal: {normal.sum()}")
    print(f"  overlap:        {(marked & normal).sum()}")
    print(f"Discarded ({rule}): {discard.sum()}")
    print(f"Kept:             {(~discard).sum()}")
    for label, count in data.loc[~discard, 'Label'].value_counts().items():
        print(f"  {label}: {count}")

    return discard


def reconcile(data, discard, mode, apply_changes):
    """Make images/ match the rule, in both directions.

    Discarded frames still in images/ are moved out; kept frames sitting in
    removed/ from an earlier, stricter run are moved back in. The second half
    is what makes it safe to loosen the rule and re-run instead of re-extracting
    the raw release. Only 'move' can restore - 'delete' is one-way.
    """
    if mode == 'move' and apply_changes:
        os.makedirs(removed_dir, exist_ok=True)

    quarantined = set(os.listdir(removed_dir)) if os.path.isdir(removed_dir) else set()

    to_remove = [n for n in data.loc[discard, 'Image'] if os.path.isfile(os.path.join(images_dir, n))]
    to_restore = [n for n in data.loc[~discard, 'Image'] if n in quarantined]

    for image_name in to_remove:
        if not apply_changes:
            continue
        if mode == 'move':
            shutil.move(os.path.join(images_dir, image_name), os.path.join(removed_dir, image_name))
        else:
            os.remove(os.path.join(images_dir, image_name))

    for image_name in to_restore:
        if apply_changes:
            shutil.move(os.path.join(removed_dir, image_name), os.path.join(images_dir, image_name))

    verb = 'Moved out' if mode == 'move' else 'Deleted'
    prefix = '' if apply_changes else 'Would have '
    print(f"\n{prefix}{verb.lower() if prefix else verb}: {len(to_remove)} images")

    if to_restore:
        print(f"{prefix}restored from removed/: {len(to_restore)} images")
    elif mode == 'delete':
        print("Note: 'delete' mode cannot restore anything - removed frames are gone.")

    return len(to_remove), len(to_restore)


def report_orphans(data):
    on_disk = {f for f in os.listdir(images_dir) if f.lower().endswith('.png')}
    orphans = sorted(on_disk - set(data['Image']))

    if orphans:
        print(f"\nOn disk but absent from INFO.csv: {len(orphans)} (left untouched)")
        for image_name in orphans[:10]:
            print(f"  - {image_name}")


def clean(args):
    data = load_info(csv_path)
    discard = flag_discards(data, args.rule)
    reconcile(data, discard, args.mode, args.apply)
    report_orphans(data)

    if args.apply:
        data[~discard].to_csv(clean_csv_path, sep=';', index=False)
        print(f"\nWrote {clean_csv_path} ({(~discard).sum()} rows)")
    else:
        print("\nDry run - nothing changed. Re-run with --apply to commit.")


def pick_val_cases(data, val_frac, seed):
    """Hold out whole cases from train until val_frac of the images is reached.

    Delegates to the helper in src/utils/dataloader.py so that splitting on disk
    and splitting at load time produce the same partition. The dataloader lives
    under a package path that cannot be imported from here (BUS-UCLM is not a
    valid identifier), so it is loaded by file path.
    """
    if val_frac <= 0:
        return []

    counts = data.groupby('case')['label'].agg(['size', lambda s: (s == '1').sum()])
    counts.columns = ['n', 'malignant']

    sizes = counts['n'].to_dict()
    malignant_counts = counts['malignant'].to_dict()

    return busuclm_pick_val_cases(sizes, sizes, malignant_counts, val_frac, seed)


def split(args):
    if not os.path.isfile(clean_csv_path):
        raise SystemExit(f"{clean_csv_path} not found - run the 'clean' stage with --apply first.")

    data = load_info(clean_csv_path)
    data['case'] = data['Image'].str[:4]
    data['label'] = data['Label'].str.lower().map(label_map)

    unmapped = data['label'].isna()
    if unmapped.any():
        raise SystemExit(f"Unexpected labels: {sorted(data.loc[unmapped, 'Label'].unique())}")

    in_test = data['case'].isin(test_cases)
    val_cases = pick_val_cases(data[~in_test], args.val_frac, args.seed)
    in_val = data['case'].isin(val_cases)

    data['split'] = 'train'
    data.loc[in_val, 'split'] = 'val'
    data.loc[in_test, 'split'] = 'test'

    missing_test = [c for c in test_cases if c not in set(data['case'])]
    if missing_test:
        print(f"Test cases with no surviving image: {', '.join(missing_test)}")
    print(f"Validation cases: {', '.join(val_cases) if val_cases else '(none)'}\n")

    summary = pd.crosstab(data['split'], data['Label'])
    summary['total'] = summary.sum(axis=1)
    print(summary.to_string(), "\n")

    for split_name in ['train', 'val', 'test']:
        cases = sorted(data.loc[data['split'] == split_name, 'case'].unique())
        print(f"{split_name:5s} {len(cases):2d} cases: {' '.join(cases)}")

    if not args.apply:
        print("\nDry run - nothing written. Re-run with --apply to commit.")
        return

    for split_name in ['train', 'val', 'test']:
        for label in ['0', '1']:
            target_dir = os.path.join(split_dir, split_name, label)
            if os.path.isdir(target_dir):
                shutil.rmtree(target_dir)
            os.makedirs(target_dir)

    copied, missing = 0, []

    for row in data.itertuples():
        source = os.path.join(images_dir, row.Image)

        if not os.path.isfile(source):
            missing.append(row.Image)
            continue

        shutil.copy(source, os.path.join(split_dir, row.split, row.label, row.Image))
        copied += 1

    print(f"\nCopied {copied} images into {split_dir}")

    if missing:
        print(f"Listed in {clean_csv_path} but not in {images_dir}: {len(missing)}")
        for image_name in missing[:10]:
            print(f"  - {image_name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='stage', required=True)

    clean_parser = subparsers.add_parser('clean', help='drop marked and Normal images')
    clean_parser.add_argument('--rule', choices=['any', 'all', 'normal', 'marks'], default='normal',
                              help="which rows to discard: 'normal' drops Label=Normal only "
                                   "(default); 'marks' drops Marks=Yes only; 'any' drops "
                                   "either; 'all' drops only rows that are both")
    clean_parser.add_argument('--mode', choices=['move', 'delete'], default='move',
                              help="'move' quarantines into removed/ (default); "
                                   "'delete' is unrecoverable")
    clean_parser.add_argument('--apply', action='store_true',
                              help='perform the removal (otherwise dry run)')
    clean_parser.set_defaults(func=clean)

    split_parser = subparsers.add_parser('split', help='build png/{train,val,test}/{0,1}')
    split_parser.add_argument('--val-frac', type=float, default=0.15,
                              help='share of the non-test images to hold out for '
                                   'validation, allocated whole cases at a time '
                                   '(default 0.15; use 0 for no validation set)')
    split_parser.add_argument('--seed', type=int, default=42,
                              help='seed for the validation case draw (default 42)')
    split_parser.add_argument('--apply', action='store_true',
                              help='write the partitions (otherwise dry run)')
    split_parser.set_defaults(func=split)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
