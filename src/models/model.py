import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ======================================================
# == 基础组件：DoubleConv1D、CrossAttention1D ==========
# ======================================================

class DoubleConv1D(nn.Module):
    """(Conv => BN => ReLU) * 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class CrossAttention1D(nn.Module):
    def __init__(self, dim, cond_dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim

        self.k_proj = nn.Linear(cond_dim, dim)
        self.v_proj = nn.Linear(cond_dim, dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim, kdim=dim, vdim=dim,
            num_heads=num_heads, batch_first=True
        )

    def forward(self, x, cond):
        """
        x: [B, C, L]
        cond: [B, N, cond_dim]
        """
        # x: [B, C, L] -> [B, L, C]
        q = x.permute(0, 2, 1)               # [B, L, dim]
        k = self.k_proj(cond)                # [B, N, dim]
        v = self.v_proj(cond)                # [B, N, dim]

        out, _ = self.attn(q, k, v)          # [B, L, dim]
        out = out.permute(0, 2, 1)           # [B, dim, L]

        return x + out                       # 残差
        
# ======================================================
# == 编码/解码模块 ======================================
# ======================================================

class EncoderBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.pool = nn.MaxPool1d(2)
        self.conv = DoubleConv1D(in_ch, out_ch)
        self.cross_attn = CrossAttention1D(out_ch, cond_dim)

    def forward(self, x, cond):
        x = self.pool(x)
        x = self.conv(x)
        x = self.cross_attn(x, cond)
        return x

class DecoderBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv = DoubleConv1D(in_ch, out_ch)
        self.cross_attn = CrossAttention1D(out_ch, cond_dim)

    def forward(self, x, skip, cond):
        x = self.up(x)
        diff = skip.shape[-1] - x.shape[-1]
        x = F.pad(x, [diff // 2, diff - diff // 2])
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        x = self.cross_attn(x, cond)
        return x

# ======================================================
# == FPN 分类头 ========================================
# ======================================================

class FPNClassifier1D(nn.Module):
    def __init__(self, feat_chs, fpn_dim=64, num_classes=3):
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv1d(ch, fpn_dim, 1) for ch in feat_chs])
        self.output_conv = nn.Conv1d(fpn_dim, fpn_dim, 3, padding=1)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(fpn_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, feats):
        # feats: [x1, x2, ..., xN] from encoder
        fpn_feats = []
        for lateral, f in zip(self.lateral_convs, feats):
            fpn_feats.append(lateral(f))
        
        f = fpn_feats[-1]
        for prev in reversed(fpn_feats[:-1]):
            f = F.interpolate(f, size=prev.shape[-1], mode='linear', align_corners=True)
            f = f + prev
        f = self.output_conv(f)
        return self.classifier(f)

# ======================================================
# == 主干模型 ==========================================
# ======================================================

class MultiTaskUNet1D(nn.Module):
    """
    完整的多任务 U-Net:
      - 输入: 1D 曲线
      - 条件: VideoMAE/I3D 特征
      - 输出: 回归强度曲线 + 分类 logits
    """
    def __init__(self, in_ch=1, base_ch=64, cond_dim=512, num_layers=3, num_classes=3):
        super().__init__()

        # 通道配置
        chs = [base_ch * (2**i) for i in range(num_layers+1)]
        self.chs = chs

        # 编码器
        self.inc = DoubleConv1D(in_ch, chs[0])
        self.encoders = nn.ModuleList([
            EncoderBlock1D(chs[i], chs[i+1], cond_dim) for i in range(num_layers)
        ])

        # 解码器
        self.decoders = nn.ModuleList([
            DecoderBlock1D(chs[i+1]+chs[i], chs[i], cond_dim) for i in reversed(range(num_layers))
        ])

        # 回归输出头
        self.reg_head = nn.Conv1d(chs[0], 1, 1)

        # 分类头 (FPN)
        self.classifier = FPNClassifier1D(feat_chs=chs[:num_layers+1], fpn_dim=base_ch, num_classes=num_classes)

    def forward(self, x, cond):

        # print(x.shape)   
        # print(cond.shape) 
        
        # 编码
        feats = []
        x = self.inc(x)
        feats.append(x)
        for enc in self.encoders:
            x = enc(x, cond)
            feats.append(x)

        # 解码
        skips = feats[:-1][::-1]
        x = feats[-1]
        for dec, skip in zip(self.decoders, skips):
            x = dec(x, skip, cond)

        # 回归输出
        reg_curve = self.reg_head(x)
        reg_curve = F.relu(reg_curve)

        # 分类输出
        cls_logits = self.classifier(feats)

        return reg_curve, cls_logits

if __name__ == "__main__":
    B, L, T, C = 2, 100, 10, 512
    curve = torch.randn(B, 1, L)         # 光流强度曲线
    cond_feat = torch.randn(B, T, C)     # 条件向量 (I3D / VideoMAE)

    model = MultiTaskUNet1D(
        in_ch=1,
        base_ch=32,
        cond_dim=512,
        num_layers=3,
        num_classes=3
    )

    reg_curve, cls_logits = model(curve, cond_feat)
    print(f"回归曲线输出: {reg_curve.shape}")  # [B, 1, L]
    print(f"分类 logits: {cls_logits.shape}") # [B, num_classes]