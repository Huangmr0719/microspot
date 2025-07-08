import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MicroSpotDataset(Dataset):
    """
    单窗口数据读取（预期配对：
     - flow_curve, micro_dhg, macro_dhg
     - videomae_feature
     - meta: subject, video, win_start, win_end, has_micro, has_macro

    返回：
      flow_curve [1,L]
      micro_dhg  [1,L]
      macro_dhg  [1,L]
      video_feat [N,D] (若存成 [D] 则自动扩维)
      cls_label  0/1/2
      meta       dict
    """

    def __init__(self, file_list, load_video_feat=True):
        super().__init__()
        # 只保留非 _videomae.npz 文件
        self.files = sorted([f for f in file_list if not f.endswith('_videomae.npz')])
        self.load_video_feat = load_video_feat

    def _match_vmae(self, flow_path: str) -> str:
        """win123.npz -> win123_videomae.npz"""
        return flow_path.replace('.npz', '_videomae.npz')

    @staticmethod
    def _label_from_meta(meta: dict) -> int:
        """
        0: none        (has_micro=False & has_macro=False)
        1: micro only
        2: macro (含 macro+micro)
        """
        m = bool(meta.get('has_micro', False))
        M = bool(meta.get('has_macro', False))
        if not m and not M: return 0
        if m and not M:     return 1
        return 2  # macro 或 macro+micro

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        flow_npz = self.files[idx]
        flow = np.load(flow_npz, allow_pickle=True)
        meta = flow['meta'].item()

        sample = {
            'flow_curve': torch.tensor(flow['flow_curve']).float().unsqueeze(0),   # [1,L]
            'micro_dhg' : torch.tensor(flow['micro_dhg']).float().unsqueeze(0),    # [1,L]
            'macro_dhg' : torch.tensor(flow['macro_dhg']).float().unsqueeze(0),    # [1,L]
            'cls_label' : torch.tensor(self._label_from_meta(meta), dtype=torch.long),
            'meta'      : meta
        }

        if self.load_video_feat:
            v_path = self._match_vmae(flow_npz)
            assert os.path.exists(v_path), f"Missing video feature: {v_path}"
            v_npz = np.load(v_path)

            # print(v_npz['videomae_feature'].shape)
            
            v_feat = torch.tensor(v_npz['videomae_feature']).float()
            if v_feat.ndim == 1:
                v_feat = v_feat.unsqueeze(0)  # → [1,D]
            sample['video_feat'] = v_feat

        return sample