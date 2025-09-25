#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np


def variants_for_key(k: str):
    vs = [k]
    if 'mask__' in k:
        vs.append(k.replace('mask__', 'mask_', 1))
    elif 'mask_' in k:
        vs.append(k.replace('mask_', 'mask__', 1))
    if ',' in k:
        vs.append(k.replace(',', ''))
    if k.endswith('_icg') and ',_icg' not in k:
        vs.append(k[:-4] + ',_icg')
    return vs


def count_presence(train_dir: str, merge_icg: bool):
    # Base classes
    base = ['mask__artery', 'mask__vein', 'mask__suture', 'mask__stroma', 'mask__umbilical_cord']
    icg = ['mask__stroma_icg', 'mask__artery_icg']
    # Classes to evaluate
    cls = base.copy()
    merge_aliases = {}
    if not merge_icg:
        cls += icg
    else:
        merge_aliases['mask__stroma'] = ['mask__stroma_icg']
        merge_aliases['mask__artery'] = ['mask__artery_icg']

    files = sorted(glob.glob(os.path.join(train_dir, '**', '*.npz'), recursive=True))
    counts = {k: 0 for k in cls}
    exist = set()
    for p in files:
        try:
            with np.load(p, allow_pickle=False) as npz:
                keys = list(npz.keys())
                for k in keys:
                    if k.startswith('mask_') or k.startswith('mask__'):
                        exist.add(k)
                for key in cls:
                    tried = set()
                    klist = variants_for_key(key)
                    for extra in merge_aliases.get(key, []):
                        klist += variants_for_key(extra)
                    found = False
                    for kk in klist:
                        if kk in tried:
                            continue
                        tried.add(kk)
                        if kk in npz:
                            m = npz[kk]
                            if m.ndim == 3 and m.shape[-1] == 1:
                                m = m[..., 0]
                            if (m > 0).any():
                                counts[key] += 1
                            found = True
                            break
        except Exception:
            continue
    return files, sorted(exist), counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-dir', type=str, required=True)
    ap.add_argument('--merge-icg', action='store_true', help='Merge *_icg masks into base classes')
    args = ap.parse_args()

    files, exist, counts = count_presence(args.train_dir, args.merge_icg)
    print(f"TRAIN_FILES {len(files)}")
    print("UNIQUE_MASK_KEYS", exist)
    print("PRESENCE_COUNTS (images with >0 pixels):")
    for k in sorted(counts.keys()):
        print(f"  - {k}: {counts[k]}")


if __name__ == '__main__':
    main()
