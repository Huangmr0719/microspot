import torch, torch.nn as nn, torch.nn.functional as F
import pytorch_lightning as pl
import os
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
from collections import defaultdict

from src.utils.postprocess import postprocess_curve, merge_intervals_by_video, compute_segment_f1

from src.models.model import MultiTaskUNet1D

class MicroSpotLit(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        lambda_cls: float = 1.0,
        lambda_smooth: float = 0.01,
        lambda_norm: float = 0.01,
        num_classes: int = 3,
        base_channels: int = 64,
        context_dim: int = 512,
        anno_file: str = "/data/users/user6/rxh/datasets/casme^2/74.220.215.205/casme^2_anno.csv"
    ):
        super().__init__()
        self.save_hyperparameters()

        # -------- model
        self.model = MultiTaskUNet1D(
            in_ch=1,
            base_ch=base_channels,
            cond_dim=context_dim,
            num_classes=num_classes
        )

        # -------- metrics
        self.train_acc = MulticlassAccuracy(num_classes)
        self.val_acc = MulticlassAccuracy(num_classes)
        self.val_f1 = MulticlassF1Score(num_classes)

        # -------- 标注数据
        if anno_file:
            import pandas as pd
            self.anno_df = pd.read_csv(anno_file)
        else:
            self.anno_df = None

    def forward(self, flow_curve, video_feat):
        # print("video_feat::::",video_feat.shape)
        return self.model(flow_curve, video_feat)

    @staticmethod
    def _smooth_l1(curve):
        return F.l1_loss(curve[:, :, 1:], curve[:, :, :-1])

    def _compute_loss(self, pred_curve, gt_curve, cls_logits, cls_label):
        reg_loss = F.mse_loss(pred_curve, gt_curve)
        cls_loss = F.cross_entropy(cls_logits, cls_label)
        smooth_loss = self._smooth_l1(pred_curve)
        norm_loss = torch.mean(torch.abs(pred_curve))
        total = (
            reg_loss +
            self.hparams.lambda_cls * cls_loss +
            self.hparams.lambda_smooth * smooth_loss +
            self.hparams.lambda_norm * norm_loss
        )
        return total, dict(reg=reg_loss, cls=cls_loss, smooth=smooth_loss, norm=norm_loss)

    def training_step(self, batch, batch_idx):
        reg_pred, cls_logits = self(batch['flow_curve'], batch['video_feat'])
        gt_curve = batch['micro_dhg'] + batch['macro_dhg']
        total, parts = self._compute_loss(reg_pred, gt_curve, cls_logits, batch['cls_label'])
        self.log_dict({f"train/{k}": v for k, v in parts.items()}, on_step=False, on_epoch=True)
        self.log("train/total", total, prog_bar=True)
        preds = torch.argmax(cls_logits, dim=1)
        self.train_acc(preds, batch['cls_label'])
        self.log("train/acc", self.train_acc, prog_bar=True)
        return total

    def on_validation_epoch_start(self):
        self.validation_windows = []
        self.saved_curves = [] 

    def validation_step(self, batch, batch_idx):
        reg_pred, cls_logits = self(batch['flow_curve'], batch['video_feat'])
        gt_curve = batch['micro_dhg'] + batch['macro_dhg']
        total, parts = self._compute_loss(reg_pred, gt_curve, cls_logits, batch['cls_label'])
        self.log_dict({f"val/{k}": v for k, v in parts.items()}, on_step=False, on_epoch=True)
        self.log("val/total", total, prog_bar=True)

        preds = torch.argmax(cls_logits, dim=1)
        self.val_acc(preds, batch['cls_label'])
        self.val_f1(preds, batch['cls_label'])

        B = reg_pred.size(0)
        for b in range(B):
            if len(self.saved_curves) < 10 and batch['cls_label'][b].item() != 0:
                pred_curve = reg_pred[b, 0].detach().cpu().numpy()
                gt = gt_curve[b, 0].detach().cpu().numpy()
                meta = batch['meta'][b]
                self.saved_curves.append(dict(pred=pred_curve, gt=gt, meta=meta))

        for b in range(reg_pred.size(0)):
            curve = reg_pred[b, 0].detach().cpu().numpy()
            logits = cls_logits[b].detach().cpu().numpy()
            meta = batch['meta'][b]
            window_result = postprocess_curve(
                intensity_curve=curve,
                cls_logits=logits,
                meta=meta,
                threshold=0.1,
                min_len=2,
                return_all_classes=True
            )
            self.validation_windows.append(window_result)

    def on_validation_epoch_end(self):
        self.log("val/acc", self.val_acc.compute(), prog_bar=True)
        self.log("val/f1", self.val_f1.compute(), prog_bar=True)
        self.val_acc.reset(); self.val_f1.reset()

        videos = defaultdict(list)
        for win in self.validation_windows:
            vid = f"{win['meta']['subject']}_{win['meta']['video']}"
            videos[vid].append(win)

        micro_f1, macro_f1 = [], []

        for vid, wins in videos.items():
            merged = merge_intervals_by_video(wins)

            gt_micro = self.get_gt_intervals(vid, 'micro')
            gt_macro = self.get_gt_intervals(vid, 'macro')

            pred_micro = [seg for seg in merged if seg['pred_class'] == 1]
            pred_macro = [seg for seg in merged if seg['pred_class'] == 2]

            score_micro = compute_segment_f1(pred_micro, gt_micro)
            score_macro = compute_segment_f1(pred_macro, gt_macro)

            micro_f1.append(score_micro['F1'])
            macro_f1.append(score_macro['F1'])

        avg_micro_f1 = sum(micro_f1) / len(micro_f1) if micro_f1 else 0.
        avg_macro_f1 = sum(macro_f1) / len(macro_f1) if macro_f1 else 0.

        self.log("val/seg_f1_micro", avg_micro_f1)
        self.log("val/seg_f1_macro", avg_macro_f1)

        os.makedirs('curve_vis', exist_ok=True)
        for i, item in enumerate(self.saved_curves):
            pred = item['pred']
            gt   = item['gt']
            meta = item['meta']
            plt.figure()
            plt.plot(pred, label='Pred Curve')
            plt.plot(gt,   label='GT Curve')
            plt.title(f"{meta['subject']} {meta['video']} win={meta['win_start']}-{meta['win_end']}")
            plt.legend()
            plt.savefig(f"curve_vis/curve_{i}.png")
            plt.close()

        self.saved_curves = []  # 重置

    def get_gt_intervals(self, video_id, mode="micro"):
        """
        根据 anno_df 提取 GT 区间 (onset, apex, offset)
        """
        if self.anno_df is None:
            return []
        sub, vid = video_id.split('_', 1)
        df = self.anno_df[(self.anno_df['subject'] == sub) & (self.anno_df['video_name'] == vid)]
        if mode == "micro":
            df = df[df['type_idx'] == 0]
        else:
            df = df[df['type_idx'] == 1]
        return [(row['start_frame'], row['apex_frame'], row['end_frame']) for _, row in df.iterrows()]

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)