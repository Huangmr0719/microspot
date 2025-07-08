import os
import glob
import random
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from .dataset import MicroSpotDataset


# ---------------- collate_fn ----------------
def microspot_collate_fn(batch):
    """
    Custom collate for Micro-Spot Dataset.
    Handles:
      - flow_curve, micro_dhg, macro_dhg : [B, 1, L]
      - video_feat : variable-length [N,D] => padded to [B, N_max, D]
      - cls_label : [B]
      - meta : list of dicts
    """
    out = {}
    keys = batch[0].keys()
    for k in keys:
        if k == 'meta':
            out[k] = [b[k] for b in batch]
        elif k == 'video_feat':
            feats = [b[k] for b in batch]
            out[k] = pad_sequence(feats, batch_first=True)  # [B, N_max, D]
        else:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


# ---------------- LightningDataModule ----------------
class MicroSpotDataModule(pl.LightningDataModule):
    """
    LightningDataModule for Micro-Spot windows.
    Supports:
      - LOSO split or random split
      - Uses MicroSpotDataset & microspot_collate_fn
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        root = self.cfg.paths.preprocess_dir
        all_npz = glob.glob(os.path.join(root, '**', 'win[0-9]*.npz'), recursive=True)
        all_npz = sorted([f for f in all_npz if not f.endswith('_videomae.npz')])

        random.seed(self.cfg.seed)

        if self.cfg.split.mode == 'LOSO':
            test_sub = self.cfg.split.test_subject
            self.test_files = [f for f in all_npz if f'/{test_sub}_' in f]
            remain = [f for f in all_npz if f not in self.test_files]
            random.shuffle(remain)
            k = int(0.1 * len(remain))
            self.val_files = remain[:k]
            self.train_files = remain[k:]
        else:
            random.shuffle(all_npz)
            n = len(all_npz)
            v = int(0.1 * n)
            self.test_files = all_npz[:2*v]
            self.val_files  = all_npz[2*v:3*v]
            self.train_files= all_npz[3*v:]

        self.ds_train = MicroSpotDataset(self.train_files, load_video_feat=True)
        self.ds_val   = MicroSpotDataset(self.val_files,   load_video_feat=True)
        self.ds_test  = MicroSpotDataset(self.test_files,  load_video_feat=True)

        print(f"✅ Loaded: train={len(self.train_files)} val={len(self.val_files)} test={len(self.test_files)}")

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size = self.cfg.loader.batch_size,
            shuffle = True,
            num_workers = self.cfg.loader.num_workers,
            pin_memory = True,
            collate_fn = microspot_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size = self.cfg.loader.batch_size,
            shuffle = False,
            num_workers = self.cfg.loader.num_workers,
            pin_memory = True,
            collate_fn = microspot_collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size = self.cfg.loader.batch_size,
            shuffle = False,
            num_workers = self.cfg.loader.num_workers,
            pin_memory = True,
            collate_fn = microspot_collate_fn
        )