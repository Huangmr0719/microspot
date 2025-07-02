import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MicroExpDataset(torch.utils.data.Dataset):
    def __init__(self, npz_list, use_flow=True, use_i3d=True):
        super().__init__()
        self.npz_list = npz_list
        self.use_flow = use_flow
        self.use_i3d = use_i3d

    def __len__(self):
        return len(self.npz_list)

    def __getitem__(self, idx):
        data = np.load(self.npz_list[idx], allow_pickle=True)
        combined_flow = data['combined_flow'][:, None]  # [T, 1]
        i3d_feat = data['i3d_feat']                      # [T, D]

        # === 拼接特征 ===
        feat = []
        if self.use_flow:
            feat.append(combined_flow)
        if self.use_i3d:
            feat.append(i3d_feat)

        x = np.concatenate(feat, axis=-1)  # [T, D_total]

        meta = data['meta'].item()
        has_exp = meta.get('has_exp', False)

        x = torch.tensor(x).float()  # [T, D]

        # === 标签，可选：回归 DHG 曲线
        if 'micro_dhg' in data:
            y = torch.tensor(data['micro_dhg']).float()
        else:
            y = torch.zeros(x.shape[0]).float()

        return x, y, has_exp, meta