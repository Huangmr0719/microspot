import hydra
from omegaconf import DictConfig
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp

# === 单滑窗处理 ===
def process_window(args):
    (frames_arr, win_start, win_end, L, W1, W2, BASE, PEAK, USE_PSEUDO,
     subject, clip_name, expr_type, output_dir, matched_event) = args

    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()

    def compute_mag(prev, next):
        flow = tvl1.calc(prev, next, None)
        return np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()

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

    init_frame = frames_arr[0]
    local_flows, cum_flows = [], []
    prev_frame = None

    for img in tqdm(frames_arr, desc="frame process"):
        if prev_frame is None:
            local_flows.append(0.0)
        else:
            local_flows.append(compute_mag(prev_frame, img))
        cum_flows.append(compute_mag(init_frame, img))
        prev_frame = img

    local_flows = np.array(local_flows)
    cum_flows = np.array(cum_flows)
    combined = W1 * local_flows + W2 * cum_flows
    eps = 1e-6
    combined_norm = (combined - combined.min()) / (combined.max() - combined.min() + eps)

    if matched_event:
        onset = matched_event['start_frame']
        apex  = matched_event['apex_frame']
        offset= matched_event['end_frame']
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
            local_peak = np.argmax(combined_norm[on_win:off_win+1]) + on_win
            ap_win = local_peak
            dhg_curve = gen_dhg(L, on_win, ap_win, off_win, BASE, PEAK)
            has_micro = expr_type == 'micro'
            has_macro = expr_type == 'macro'
        else:
            dhg_curve = np.zeros(L)
            ap_win = -1; on_win = -1; off_win = -1
            has_micro = False; has_macro = False
    else:
        dhg_curve = np.zeros(L)
        ap_win = -1; on_win = -1; off_win = -1
        has_micro = False; has_macro = False

    np.savez(
        os.path.join(output_dir, f"{subject}_{clip_name}_win{win_start}.npz"),
        local_flow=local_flows.astype(np.float32),
        cum_flow=cum_flows.astype(np.float32),
        combined_flow=combined_norm.astype(np.float32),
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
    L = cfg.preprocess.window_size
    S = cfg.preprocess.stride
    W1 = cfg.preprocess.flow.w1
    W2 = cfg.preprocess.flow.w2
    BASE = cfg.preprocess.dhg.base
    PEAK = cfg.preprocess.dhg.peak
    USE_PSEUDO = cfg.preprocess.pseudo_apex.use_if_overlap_half
    output_dir = cfg.preprocess.output_dir
    os.makedirs(output_dir, exist_ok=True)

    anno_df = pd.read_csv(cfg.paths.anno_file)
    video_groups = anno_df.groupby('video_name')

    for clip_name, group in tqdm(video_groups, desc='Video Groups'):
        first_row = group.iloc[0]
        sub_id = int(first_row['subject'])
        subject = f"s{sub_id:02d}"

        frames_dir = os.path.join(cfg.preprocess.input_dir, subject, clip_name)
        frames = sorted([
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir) if f.endswith('.jpg')
        ])

        if len(frames) < L:
            continue

        frames_arr = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in frames]
        n_frames = len(frames)
        args_list = []

        for start in range(0, n_frames - L + 1, S):
            end = start + L
            win_start, win_end = start, end

            max_iou = 0; matched_event = None
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

            args_list.append((
                frames_arr[start:end], win_start, win_end, L, W1, W2, BASE, PEAK, USE_PSEUDO,
                subject, clip_name, expr_type, output_dir, matched_event
            ))

        with mp.Pool(processes=cfg.preprocess.num_workers) as pool:
            list(tqdm(pool.imap(process_window, args_list), total=len(args_list),
                      desc=f"Sliding windows: {subject}/{clip_name}"))

if __name__ == "__main__":
    main()