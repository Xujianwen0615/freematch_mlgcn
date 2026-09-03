# -*- coding: utf-8 -*-
"""
Multi-label semi-supervised DataLoader for the private dataset.

Adapted from the prior working implementation in
D:\\freematch_mlgcn\\semilearn\\datasets\\custom_semi.py.

Data layout (D:\\CV\\custom_muti)
------------------------------
    data/
        train/    <image files>
        val/      <image files>
        category.json      {"class_name": index, ...}   (C classes)
        train_anno.json    [{"file_name": ..., "labels": [indices]}, ...]
        val_anno.json      [{"file_name": ..., "labels": [indices]}, ...]
    custom_adj.pkl                {"P": [C, C]}   label correlation
    custom_glove_word2vec.pkl     [C, 300]        GloVe class embeddings

Key design
----------
  * The training annotations cover ALL training images. To simulate semi-supervised learning we
    HIDE the labels of a fraction ``1 - labeled_ratio`` of the images (standard SSL protocol).
  * ``split_mode``:
      - 'random'     → shuffle the whole train set, first ``labeled_ratio`` fraction = labeled.
      - 'by_prefix'  → split within each filename-prefix group so that every prefix contributes
                       labeled + unlabeled images (better for multi-source datasets).
  * Targets are returned as a binarized float vector in {0, 1} of length C, which is what both
    ASL and BCE expect.
"""
import json
import os
import random

import numpy as np
import pickle
import torch
from PIL import Image
from torch.utils.data import Dataset


class MultiLabelDataset(Dataset):
    def __init__(self, data_dir, phase="train", weak_transform=None, strong_transform=None,
                 is_unlabeled=False, labeled_ratio=0.2, seed=42, split_mode="random"):
        self.data_dir = data_dir
        self.phase = phase
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        self.is_unlabeled = is_unlabeled
        self.labeled_ratio = labeled_ratio
        self.seed = seed
        self.split_mode = split_mode

        if phase == "train":
            anno_path = os.path.join(data_dir, "data", "train_anno.json")
            assert os.path.exists(anno_path), f"train annotation not found: {anno_path}"
            with open(anno_path, "r", encoding="utf-8") as f:
                full = json.load(f)

            random.seed(seed)

            if split_mode == "by_prefix":
                groups = {}
                for item in full:
                    prefix = item["file_name"].split("-")[0]
                    groups.setdefault(prefix, []).append(item)
                lb, ulb = [], []
                for prefix in sorted(groups.keys()):
                    g = groups[prefix]
                    random.shuffle(g)
                    split_idx = max(1, int(len(g) * labeled_ratio))
                    lb.extend(g[:split_idx])
                    ulb.extend(g[split_idx:])
            else:  # random
                random.shuffle(full)
                split_idx = int(len(full) * labeled_ratio)
                lb, ulb = full[:split_idx], full[split_idx:]

            self.img_list = ulb if is_unlabeled else lb
            total = len(full)
            print(f"[data] {phase} {('unlabeled' if is_unlabeled else 'labeled')} "
                  f"subset: {len(self.img_list)} / {total} "
                  f"({len(self.img_list) / total * 100:.1f}%)")
        else:
            anno_path = os.path.join(data_dir, "data", f"{phase}_anno.json")
            assert os.path.exists(anno_path), f"{phase} annotation not found: {anno_path}"
            with open(anno_path, "r", encoding="utf-8") as f:
                self.img_list = json.load(f)
            print(f"[data] {phase} set: {len(self.img_list)} samples")

        cat_path = os.path.join(data_dir, "data", "category.json")
        assert os.path.exists(cat_path), f"category map not found: {cat_path}"
        with open(cat_path, "r", encoding="utf-8") as f:
            self.cat2idx = json.load(f)
        self.num_classes = len(self.cat2idx)
        self.idx2cat = {v: k for k, v in self.cat2idx.items()}

        self.inp = None
        self.inp_file = None

    def load_class_embeddings(self, inp_file):
        """Load GloVe-style [C, d] embeddings once (used by the model, kept on the dataset)."""
        self.inp_file = inp_file
        if inp_file is not None and os.path.exists(inp_file):
            with open(inp_file, "rb") as f:
                self.inp = np.asarray(pickle.load(f), dtype=np.float32)
            print(f"[data] loaded class embeddings: {self.inp.shape}")
        elif inp_file is not None:
            print(f"[data] WARNING: embedding file not found {inp_file}; using random init.")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        item = self.img_list[index]
        img_folder = os.path.join(self.data_dir, "data", self.phase)
        img_path = os.path.join(img_folder, item["file_name"])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")

        # Unlabeled: return weak + strong views.
        if self.phase == "train" and self.is_unlabeled:
            weak = self.weak_transform(img) if self.weak_transform is not None else img
            strong = self.strong_transform(img) if self.strong_transform is not None else img
            return weak, strong

        # Labeled / val: return image + binarized {0,1} target.
        img = self.weak_transform(img) if self.weak_transform is not None else img
        target = np.zeros(self.num_classes, np.float32)
        target[item["labels"]] = 1.0
        return img, target


def collate_fn(batch):
    """Pack a batch. Labeled/val -> (images, targets); unlabeled -> (weak, strong)."""
    if len(batch) == 0:
        raise ValueError("empty batch")
    first = batch[0]
    if isinstance(first, (tuple, list)) and len(first) == 2 and isinstance(first[1], torch.Tensor) \
            and first[1].ndim == 3:
        # Unlabeled: (weak_img, strong_img)
        weak = torch.stack([b[0] for b in batch])
        strong = torch.stack([b[1] for b in batch])
        return weak, strong
    else:
        imgs = torch.stack([b[0] for b in batch])
        targets = torch.from_numpy(np.stack([b[1] for b in batch])).float()
        return imgs, targets


if __name__ == "__main__":
    from torchvision import transforms
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=r"D:\CV\custom_muti")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--ratio", type=float, default=0.2)
    p.add_argument("--inp", default=r"D:\CV\custom_muti\custom_glove_word2vec.pkl")
    a = p.parse_args()

    tf = transforms.Compose([transforms.Resize((a.img_size, a.img_size)),
                             transforms.ToTensor(),
                             transforms.Normalize([0.2318, 0.2274, 0.2761],
                                                  [0.2077, 0.2220, 0.2262])])
    lb = MultiLabelDataset(a.data_dir, "train", weak_transform=tf, is_unlabeled=False,
                           labeled_ratio=a.ratio, split_mode="by_prefix")
    lb.load_class_embeddings(a.inp)
    ulb = MultiLabelDataset(a.data_dir, "train", weak_transform=tf, strong_transform=tf,
                            is_unlabeled=True, labeled_ratio=a.ratio, split_mode="by_prefix")
    val = MultiLabelDataset(a.data_dir, "val", weak_transform=tf, is_unlabeled=False, labeled_ratio=1.0)

    img, tgt = lb[0]
    print("labeled sample:", img.shape, tgt.shape, "pos:", tgt.nonzero()[0].tolist())
    w, s = ulb[0]
    print("unlabeled sample:", w.shape, s.shape)
    print("num_classes:", lb.num_classes, "embeddings:", lb.inp.shape)
