from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os
from glob import glob

class MicroExpDataModule(pl.LightningDataModule):
    def __init__(self, data_root, loso_subject='s15', batch_size=8):
        super().__init__()
        self.data_root = data_root
        self.loso_subject = loso_subject
        self.batch_size = batch_size

    def setup(self, stage=None):
        all_npz = glob(os.path.join(self.data_root, "*.npz"))
        train_list, test_list = [], []

        for f in all_npz:
            meta = np.load(f, allow_pickle=True)['meta'].item()
            if meta['subject'] == self.loso_subject:
                test_list.append(f)
            else:
                train_list.append(f)

        self.train_dataset = MicroExpDataset(train_list)
        self.val_dataset   = MicroExpDataset(test_list)
        self.test_dataset  = MicroExpDataset(test_list)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=4, pin_memory=True)