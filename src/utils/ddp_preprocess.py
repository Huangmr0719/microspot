import hydra
from omegaconf import DictConfig
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor

def init_worker():
    global tvl1
    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()

def extract_frame_number(filename):
    basename = os.path.basename(filename)
    match = re.search(r'img(\d+)\.jpg', basename)
    return int(match.group(1)) if match else 0

# === 计算光流和处理单个视频 ===
def process_video(cfg, clip_name, group):
    L = cfg.preprocess.window_size
    S = cfg.preprocess.stride
    W1 = cfg.preprocess.flow.w1
    W2 = cfg.preprocess.flow.w2
    BASE = cfg.preprocess.dhg.base
    PEAK = cfg.preprocess.dhg.peak
    USE_PSEUDO = cfg.preprocess.pseudo_apex.use_if_overlap_half
    output_dir = cfg.preprocess.output_dir
    
    first_row = group.iloc[0]
    sub_id = int(first_row['subject'])
    subject = f"s{sub_id:02d}"

    # 创建视频专属子文件夹
    video_output_dir = os.path.join(output_dir, f"{subject}_{clip_name}")
    os.makedirs(video_output_dir, exist_ok=True)

    frames_dir = os.path.join(cfg.preprocess.input_dir, subject, clip_name)
    frames = sorted(
        [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.jpg')],
        key=extract_frame_number
    )

    if len(frames) < L:
        return

    frames_arr = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in frames]
    n_frames = len(frames_arr)
    
    local_flows = np.zeros(n_frames, dtype=np.float32)
    cum_flows = np.zeros(n_frames, dtype=np.float32)
    init_frame = frames_arr[0]
    
    for i in tqdm(range(1, n_frames), desc=f"计算光流: {subject}/{clip_name}"):
        prev_frame = frames_arr[i-1]
        curr_frame = frames_arr[i]
        
        # 计算局部光流
        flow = tvl1.calc(prev_frame, curr_frame, None)
        local_flows[i] = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()
        
        # 计算累积光流
        flow = tvl1.calc(init_frame, curr_frame, None)
        cum_flows[i] = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()
    
    eps = 1e-6
    combined = W1 * local_flows + W2 * cum_flows
    combined_norm = (combined - combined.min()) / (combined.max() - combined.min() + eps)
    
    # === 处理所有窗口 ===
    for start in range(0, n_frames - L + 1, S):
        end = start + L
        win_start, win_end = start, end

        max_iou = 0
        matched_event = None
        for _, row in group.iterrows():
            onset = int(row['start_frame'])
            offset = int(row['end_frame'])
            iou = max(0, min(win_end, offset) - max(win_start, onset)) / \
                  (max(win_end, offset) - min(win_start, onset) + 1e-6)
            if iou > max_iou:
                max_iou = iou
                matched_event = row.to_dict()

        if max_iou <= 0:
            matched_event = None

        expr_type = 'macro' if (matched_event and matched_event['type_idx'] == 1) else 'micro'

        # === 窗口内处理逻辑 ===
        def get_on_off_win(onset, offset, win_start, win_end, L):
            if onset < win_start:
                on_win = 0
            else:
                on_win = max(onset - win_start, 0)
            if offset >= win_end:
                off_win = L - 1
            else:
                off_win = max(min(offset, win_end) - win_start, 0)
            return on_win, off_win

        def gen_dhg(L, on_win, ap_win, off_win, BASE, PEAK):
            sigma_l = max((ap_win - on_win) / 3.0, 1.0)
            sigma_r = max((off_win - ap_win) / 3.0, 1.0)
            curve = np.zeros(L)
            for i in range(on_win, off_win + 1):
                delta = (i - ap_win) / sigma_l if i <= ap_win else (i - ap_win) / sigma_r
                curve[i] = BASE + (PEAK - BASE) * np.exp(-0.5 * delta**2)
            curve[0] = 0.0
            curve[on_win] = BASE; curve[ap_win] = PEAK; curve[off_win] = BASE; curve[-1] = 0.0
            if on_win > 0:
                curve[:on_win] = np.linspace(0.0, BASE, on_win, endpoint=False)
            if off_win + 1 < L:
                curve[off_win + 1:] = np.linspace(BASE, 0.0, L - (off_win + 1))
            return curve

        if matched_event:
            onset = matched_event['start_frame']
            apex = matched_event['apex_frame']
            offset = matched_event['end_frame']
            type_idx = matched_event['type_idx']
            expr_type = 'macro' if type_idx == 1 else 'micro'

            covers_apex = apex >= win_start and apex < win_end
            overlap_len = max(0, min(win_end, offset) - max(win_start, onset))
            on_win, off_win = get_on_off_win(onset, offset, win_start, win_end, L)

            if covers_apex:
                ap_win = apex - win_start
                dhg_curve = gen_dhg(L, on_win, ap_win, off_win, BASE, PEAK)
                has_micro = expr_type == 'micro'
                has_macro = expr_type == 'macro'
            elif overlap_len >= L // 2 and USE_PSEUDO:
                local_peak = np.argmax(combined_norm[start:end][on_win:off_win+1]) + on_win
                ap_win = local_peak
                dhg_curve = gen_dhg(L, on_win, ap_win, off_win, BASE, PEAK)
                has_micro = expr_type == 'micro'
                has_macro = expr_type == 'macro'
            else:
                dhg_curve = np.zeros(L)
                ap_win = -1; on_win = -1; off_win = -1;
                has_micro = False; has_macro = False
        else:
            dhg_curve = np.zeros(L)
            ap_win = -1; on_win = -1; off_win = -1;
            has_micro = False; has_macro = False

        # 保存结果到视频专属子文件夹
        np.savez(
            os.path.join(video_output_dir, f"win{win_start}.npz"),
            local_flow=local_flows[start:end].astype(np.float32),
            cum_flow=cum_flows[start:end].astype(np.float32),
            combined_flow=combined_norm[start:end].astype(np.float32),
            micro_dhg=dhg_curve if has_micro else np.zeros(L),
            macro_dhg=dhg_curve if has_macro else np.zeros(L),
            meta=dict(
                subject=subject,
                clip=clip_name,
                expr_type=expr_type,
                win_start=win_start,
                win_end=win_end,
                on_win=int(on_win),
                ap_win=int(ap_win),
                off_win=int(off_win),
                has_micro=has_micro,
                has_macro=has_macro
            )
        )

@hydra.main(config_path="/data/users/user6/hmr/code/microspot/configs", config_name="preprocess.yaml")
def main(cfg: DictConfig):
    # 读取注释文件
    anno_df = pd.read_csv(cfg.paths.anno_file)
    video_groups = anno_df.groupby('video_name')
    
    # 创建主输出目录
    output_dir = cfg.preprocess.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 按视频任务分配进程
    with ProcessPoolExecutor(max_workers=cfg.preprocess.num_workers, initializer=init_worker) as executor:
        # 使用partial绑定cfg参数
        process_func = partial(process_video, cfg)
        list(tqdm(executor.map(process_func, video_groups.groups.keys(), 
                              [video_groups.get_group(name) for name in video_groups.groups.keys()]),
                 total=len(video_groups), desc='处理视频'))

if __name__ == "__main__":
    main()    