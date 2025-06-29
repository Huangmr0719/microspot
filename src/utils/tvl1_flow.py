# from pathlib import Path
# import glob
# import re, cv2
# import numpy as np

# NUM_RE = re.compile(r'(\d+)') 

# def natural_key(path: Path):
#     """按文件名中的整数自然排序."""
#     match = NUM_RE.search(path.stem)
#     if not match:
#         raise ValueError(f'File {path.name} lacks an integer index.')
#     return int(match.group(1))

# def load_sequence(seq_dir: str | Path, check_contiguous: bool = True):
#     """
#     读取某一段微表情序列（EPxx_xx），返回灰度 float32 帧列表，长度 T。
#     要求文件名形如 img1.jpg, img2.jpg, ...
#     """
#     seq_dir = Path(seq_dir)
#     files   = sorted(seq_dir.glob('*.jpg'), key=natural_key)

#     # === 连续性检查 ===
#     if check_contiguous:
#         indices = [natural_key(f) for f in files]
#         expect  = list(range(1, len(files)+1))
#         if indices != expect:
#             missing = set(expect) - set(indices)
#             raise RuntimeError(
#                 f'Frame numbering error in {seq_dir}.\n'
#                 f'Expected {expect[0]}..{expect[-1]}, missing: {sorted(missing)}'
#             )

#     # === 读取 ===
#     frames = [
#         cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
#         for f in files
#     ]
#     return frames      # List[np.ndarray] (H,W)=224

# # ---------------------------------------------------------
# # 1. 初始化 TV-L1 光流对象
# # ---------------------------------------------------------
# def create_tvl1():
#     tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
#     # 可调参数（默认已适用微表情，高精度时再细调）
#     # tvl1.setTau(0.25)          # 时间步
#     # tvl1.setLambda(0.15)       # smooth 正则
#     # tvl1.setTheta(0.3)         # 迭代步长
#     # tvl1.setScalesNumber(5)    # 金字塔层
#     # tvl1.setWarpingsNumber(5)  # 每层 warps
#     # tvl1.setEpsilon(0.01)      # 终止阈
#     # tvl1.setIterations(300)    # 迭代数
#     return tvl1

# # ---------------------------------------------------------
# # 2. 逐帧计算光流 & 幅度
# # ---------------------------------------------------------
# def tvl1_flow_magnitude(sequence, tvl1=None):
#     if tvl1 is None:
#         tvl1 = create_tvl1()
#     mags, flows = [], []
#     for i in range(1, len(sequence)):
#         prev, nxt = sequence[i-1], sequence[i]
#         flow = tvl1.calc(prev, nxt, None)      # shape (H,W,2)
#         u, v = flow[..., 0], flow[..., 1]
#         mag = np.sqrt(u**2 + v**2)             # 像素级幅度
#         mags.append(mag)                       # (H,W)
#         flows.append(flow)                     # 保存向量场 (可选)
#     return mags, flows

# # ---------------------------------------------------------
# # 3. 获得时序强度曲线
# #    这里取人脸区域平均幅度。若想突出局部，可做
# #    Top-k pooling 或 ROI 统计。
# # ---------------------------------------------------------
# def magnitude_curve(mags, pool='mean'):
#     if pool == 'mean':
#         return np.array([m.mean() for m in mags])
#     elif pool == 'max':
#         return np.array([m.max() for m in mags])
#     elif pool == 'topk':         # 拿幅度前 10% 均值
#         k = int(0.1 * mags[0].size)
#         return np.array([np.sort(m.flatten())[-k:].mean() for m in mags])
#     else:
#         raise ValueError

# # ------------- 例子 -------------
# seq_dir = './CASME2_sub01_ep01/'
# frames  = load_sequence(seq_dir)
# mags, flows = tvl1_flow_magnitude(frames)
# curve = magnitude_curve(mags, pool='mean')

# curve_norm = (curve - curve.min()) / (curve.ptp() + 1e-6)

from pathlib import Path
import re, cv2, numpy as np
from tqdm import tqdm 

SRC_ROOT = Path('/data/users/user6/rxh/datasets/casme^2/74.220.215.205/Cropped224_all')
DST_ROOT = Path('/data/users/user6/rxh/datasets/casme^2/74.220.215.205/casme^2_curves')         
DST_ROOT.mkdir(parents=True, exist_ok=True)

NUM_RE = re.compile(r'(\d+)')
def natural_key(p: Path) -> int:
    m = NUM_RE.search(p.stem)
    if not m:
        raise ValueError(f'{p} lacks integer index')
    return int(m.group(1))

def contiguous_files(files):
    idx = [natural_key(f) for f in files]
    expect = list(range(1, len(files)+1))
    if idx != expect:
        miss = sorted(set(expect)-set(idx))
        raise RuntimeError(f'Frame missing: {miss}')
    return files

def load_sequence(seq_dir: Path):
    files = sorted(seq_dir.glob('*.jpg'), key=natural_key)
    files = contiguous_files(files)
    return [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0
            for f in files]

def create_tvl1():
    return cv2.optflow.DualTVL1OpticalFlow_create()

def tvl1_curve(frames, tvl1):
    mags = []
    iterator = range(1, len(frames))
    iterator = tqdm(iterator,
                        desc='frames',
                        leave=False,
                        position=1,
                        unit='f')
    for i in iterator:
        flow = tvl1.calc(frames[i-1], frames[i], None)
        mag  = np.sqrt((flow[...,0]**2)+(flow[...,1]**2)).mean()
        mags.append(mag)
    mags = np.asarray(mags, dtype=np.float32)
    # 归一到 [0,1]
    return (mags - mags.min()) / (mags.ptp() + 1e-6)

# ---------- 主循环 ----------
tvl1 = create_tvl1()

seq_dirs = sorted([p for p in SRC_ROOT.rglob('*') if p.is_dir() and
                   any(p.glob('img*.jpg'))])

for seq in tqdm(seq_dirs, desc='processing'):
    rel_path = seq.relative_to(SRC_ROOT)             
    out_dir  = DST_ROOT / rel_path
    out_dir.mkdir(parents=True, exist_ok=True)
    curve_npy = out_dir/'curve.npy'
    if curve_npy.exists():        
        continue

    try:
        frames = load_sequence(seq)
        curve  = tvl1_curve(frames, tvl1)
    except Exception as e:
        print(f'[ERROR] {seq}: {e}')
        continue

    np.save(curve_npy, curve)

    np.savetxt(out_dir/'curve.csv', curve, delimiter=',')
    import matplotlib.pyplot as plt; plt.plot(curve); plt.savefig(out_dir/'curve.png'); plt.close()

print('✓ Done. Curves saved to', DST_ROOT)