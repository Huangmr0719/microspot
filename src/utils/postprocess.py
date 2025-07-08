import numpy as np
import torch
from scipy.ndimage import label

def softmax(x):
    x = np.asarray(x)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x) + 1e-8)

def postprocess_curve(
    intensity_curve, 
    cls_logits, 
    meta, 
    threshold=0.2, 
    min_len=2, 
    return_all_classes=False
):
    """
    单窗口后处理：强度曲线 -> (onset, apex, offset) 区间，支持多段；分类标签
    :param intensity_curve: (L,) numpy array, 输出强度曲线
    :param cls_logits: (3,) numpy/torch, 分类logits，0=none, 1=micro, 2=macro
    :param meta: dict, 必须包含 'win_start'
    :param threshold: float, 曲线阈值
    :param min_len: int, 最小持续帧数
    :param return_all_classes: bool, True则分别输出micro/macro区间
    :return: dict，包括 interval列表/类别/置信度
    """
    # 分类
    if isinstance(cls_logits, torch.Tensor):
        cls_logits = cls_logits.detach().cpu().numpy()
    prob = softmax(cls_logits)
    pred_class = int(np.argmax(prob))
    pred_score = float(np.max(prob))

    # 曲线归一化与二值化
    intensity = np.asarray(intensity_curve).copy()
    norm_intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)
    thres = max(threshold, 0.1 * norm_intensity.max())
    binary_curve = (norm_intensity > thres).astype(np.uint8)

    # 区间提取
    labeled, n_segs = label(binary_curve)
    intervals = []
    win_start = meta['win_start']
    for seg_id in range(1, n_segs + 1):
        indices = np.where(labeled == seg_id)[0]
        if len(indices) < min_len:
            continue
        onset = indices[0]
        offset = indices[-1]
        apex = onset + np.argmax(intensity[onset:offset+1])
        intervals.append(dict(
            onset=win_start + onset,
            apex=win_start + apex,
            offset=win_start + offset
        ))

    # 可选：分别处理 micro/macro_dhg 曲线（如模型多输出）
    extra = {}
    if return_all_classes:
        for key in ['micro_dhg', 'macro_dhg']:
            if key in meta:
                raw = meta[key]
                norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-6)
                bin_ = (norm > thres).astype(np.uint8)
                lb, ns = label(bin_)
                segs = []
                for seg in range(1, ns + 1):
                    idx = np.where(lb == seg)[0]
                    if len(idx) < min_len: continue
                    seg_on = idx[0]
                    seg_off = idx[-1]
                    seg_apx = seg_on + np.argmax(raw[seg_on:seg_off+1])
                    segs.append(dict(
                        onset=win_start + seg_on,
                        apex=win_start + seg_apx,
                        offset=win_start + seg_off
                    ))
                extra[key] = segs

    return {
        'intervals': intervals,         # [dict, ...]
        'pred_class': pred_class,       # 0=none, 1=micro, 2=macro
        'score': pred_score,
        'meta': meta,
        **extra
    }

# ==== 批量后处理，合并全视频 ====
def merge_intervals_by_video(results, video_len=None, mode='union', min_gap=2):
    """
    支持将多滑窗结果合并成完整视频区间（去重/合并重叠）
    :param results: list of {intervals, pred_class, meta, ...}
    :param video_len: int, 全视频总帧数
    :param mode: str, union=合并重叠区间, 'vote'=窗口多数投票
    :param min_gap: int, 相邻区间最大gap，<=则合并
    :return: 合并后区间列表
    """
    # 合并所有区间
    all_intervals = []
    for r in results:
        for seg in r['intervals']:
            all_intervals.append({'onset': seg['onset'],
                                'apex': seg['apex'],
                                'offset': seg['offset'],
                                'pred_class': r['pred_class']} )
    # 按onset排序，合并重叠
    all_intervals = sorted(all_intervals, key=lambda x: x['onset'])
    merged = []
    for seg in all_intervals:
        if not merged or seg['onset'] > merged[-1]['offset'] + min_gap:
            merged.append(seg)
        else:
            last = merged.pop()
            new = {
                'onset': min(last['onset'], seg['onset']),
                'offset': max(last['offset'], seg['offset']),
                'apex': int((last['apex'] + seg['apex']) / 2),
                'pred_class': seg['pred_class']
            }
            merged.append(new)
    return merged

import numpy as np

def compute_iou(pred_seg, gt_seg):
    """
    计算单个预测区间与 GT 区间的 IoU (Intersection over Union)
    pred_seg, gt_seg: [onset, apex, offset] 格式
    """
    pred_start, _, pred_end = pred_seg
    gt_start, _, gt_end = gt_seg

    inter_start = max(pred_start, gt_start)
    inter_end   = min(pred_end, gt_end)
    intersection = max(0, inter_end - inter_start + 1)

    union = max(pred_end, gt_end) - min(pred_start, gt_start) + 1

    return intersection / union if union > 0 else 0.0


def compute_segment_f1(
    pred_intervals,
    gt_intervals,
    iou_threshold=0.05
):
    TP = 0
    matched_gt = set()

    for pred in pred_intervals:
        for i, gt in enumerate(gt_intervals):
            iou = compute_iou(pred, gt)
            if iou >= iou_threshold and i not in matched_gt:
                TP += 1
                matched_gt.add(i)
                break

    FP = len(pred_intervals) - TP
    FN = len(gt_intervals) - TP

    P = TP / (TP + FP + 1e-8)
    R = TP / (TP + FN + 1e-8)
    F1 = 2 * P * R / (P + R + 1e-8)

    return {
        "TP": TP, "FP": FP, "FN": FN,
        "Precision": P, "Recall": R, "F1": F1
    }

# ==== 用法示例 ====
if __name__ == "__main__":
    # 假设你有模型输出:
    # pred_curve: (L,)
    # pred_logits: (3,)
    # meta = dict(win_start=100, ...)
    # 直接单窗口后处理:
    curve = np.random.rand(100)
    logits = np.random.randn(3)
    meta = {'win_start': 200}
    res = postprocess_curve(curve, logits, meta)
    print(res)

    # 合并多个窗口结果
    batch_results = [postprocess_curve(np.random.rand(100), np.random.randn(3), {'win_start': i*50}) for i in range(10)]
    merged = merge_intervals_by_video(batch_results, video_len=600)
    print("Merged intervals:", merged)