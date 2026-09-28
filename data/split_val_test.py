#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Split one third of the physical validation set into a prefix-stratified test set.

The existing training set is left untouched. Validation records are grouped by the filename
prefix before the first ``-`` (for example ``01`` in ``01-xxx.jpg``), then each group contributes
approximately one third of its records to the test set. Largest-remainder allocation makes the
global test size exactly one third of the original validation set.
"""
import argparse
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path


def _prefix(file_name):
    return file_name.split("-", 1)[0]


def _allocate_test_counts(groups, test_total, fraction):
    """Allocate an exact test total across prefix groups as proportionally as possible."""
    raw = {name: len(records) * fraction for name, records in groups.items()}
    counts = {name: math.floor(value) for name, value in raw.items()}
    remaining = test_total - sum(counts.values())
    order = sorted(groups, key=lambda name: (-(raw[name] - counts[name]), name))
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def split_validation_set(data_root, seed=42):
    data_root = Path(data_root).resolve()
    val_dir = (data_root / "val").resolve()
    test_dir = (data_root / "test").resolve()
    val_anno = (data_root / "val_anno.json").resolve()
    test_anno = (data_root / "test_anno.json").resolve()

    if not val_dir.is_dir() or not val_anno.is_file():
        raise FileNotFoundError(f"validation directory or annotations missing under {data_root}")
    if test_anno.exists() or (test_dir.exists() and any(test_dir.iterdir())):
        raise FileExistsError("test set already exists; refusing to split the validation set twice")

    records = json.loads(val_anno.read_text(encoding="utf-8"))
    if len(records) < 3:
        raise ValueError("validation set must contain at least three records")

    names = [record["file_name"] for record in records]
    if len(names) != len(set(names)):
        raise ValueError("val_anno.json contains duplicate file names")
    missing = [name for name in names if not (val_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} annotated validation images are missing; first: {missing[0]}")

    groups = defaultdict(list)
    for record in records:
        groups[_prefix(record["file_name"])].append(record)

    test_total = round(len(records) / 3)
    test_counts = _allocate_test_counts(groups, test_total, 1 / 3)
    rng = random.Random(seed)
    selected_names = set()
    for name in sorted(groups):
        group = list(groups[name])
        rng.shuffle(group)
        selected_names.update(record["file_name"] for record in group[:test_counts[name]])

    val_records = [record for record in records if record["file_name"] not in selected_names]
    test_records = [record for record in records if record["file_name"] in selected_names]
    if len(test_records) != test_total or len(val_records) + len(test_records) != len(records):
        raise RuntimeError("internal split-size mismatch")

    val_tmp = data_root / ".val_anno.json.tmp"
    test_tmp = data_root / ".test_anno.json.tmp"
    for path, payload in ((val_tmp, val_records), (test_tmp, test_records)):
        with path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")

    test_dir.mkdir(exist_ok=True)
    moved = []
    try:
        for record in test_records:
            src = val_dir / record["file_name"]
            dst = test_dir / record["file_name"]
            shutil.move(str(src), str(dst))
            moved.append((src, dst))
        test_tmp.replace(test_anno)
        val_tmp.replace(val_anno)
    except Exception:
        if test_anno.exists():
            test_anno.unlink()
        for src, dst in reversed(moved):
            if dst.exists():
                shutil.move(str(dst), str(src))
        val_tmp.unlink(missing_ok=True)
        test_tmp.unlink(missing_ok=True)
        if test_dir.exists() and not any(test_dir.iterdir()):
            test_dir.rmdir()
        raise

    per_prefix = {
        name: {"val": len(groups[name]) - test_counts[name], "test": test_counts[name]}
        for name in sorted(groups)
    }
    return len(val_records), len(test_records), per_prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="dataset/custom_muti/data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    val_count, test_count, per_prefix = split_validation_set(args.data_root, args.seed)
    print(f"split complete: val={val_count}, test={test_count}")
    for prefix, counts in per_prefix.items():
        print(f"  {prefix}: val={counts['val']}, test={counts['test']}")


if __name__ == "__main__":
    main()
