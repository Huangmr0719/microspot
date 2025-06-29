import hydra
from omegaconf import DictConfig
import os

@hydra.main(config_path="/data/users/user6/hmr/code/microspot/configs", config_name="preprocess.yaml")
def main(cfg: DictConfig):
    import cv2
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    L = cfg.preprocess.window_size
    S = cfg.preprocess.stride
    W1 = cfg.preprocess.flow.w1
    W2 = cfg.preprocess.flow.w2
    BASE = cfg.preprocess.dhg.base  # 建议 = 0.1
    PEAK = cfg.preprocess.dhg.peak  # 建议 = 1.0
    USE_PSEUDO = cfg.preprocess.pseudo_apex.use_if_overlap_half

    input_dir = cfg.preprocess.input_dir
    output_dir = cfg.preprocess.output_dir

    os.makedirs(output_dir, exist_ok=True)

    anno_df = pd.read_csv(cfg.paths.anno_file)

    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()

    def compute_mag(prev, next):
        flow = tvl1.calc(prev, next, None)
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean()
        return mag

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

    def gen_dhg(L, on_win, ap_win, off_win, BASE=0.1, PEAK=1.0):
        sigma_l = max((ap_win - on_win) / 3.0, 1.0)
        sigma_r = max((off_win - ap_win) / 3.0, 1.0)

        curve = np.zeros(L)

        for i in range(on_win, off_win + 1):
            delta = (i - ap_win) / sigma_l if i <= ap_win else (i - ap_win) / sigma_r
            curve[i] = BASE + (PEAK - BASE) * np.exp(-0.5 * delta**2)

        curve[0] = 0.0
        curve[on_win] = BASE
        curve[ap_win] = PEAK
        curve[off_win] = BASE
        curve[-1] = 0.0

        # === 渐入渐出 ===
        if on_win > 0:
            curve[:on_win] = np.linspace(0.0, BASE, on_win, endpoint=False)
        if off_win + 1 < L:
            curve[off_win + 1:] = np.linspace(BASE, 0.0, L - (off_win + 1))

        return curve

    for _, row in tqdm(anno_df.iterrows(), total=len(anno_df)):
        sub_id = int(row['subject'])
        subject = f"s{sub_id:02d}"
        clip_name = row['video_name']
        type_idx = int(row['type_idx'])
        expr_type = 'macro' if type_idx == 1 else 'micro'

        onset = int(row['start_frame'])
        apex = int(row['apex_frame'])
        offset = int(row['end_frame'])
        if offset == 0 and apex > 0:
            offset = apex + 1

        frames_dir = os.path.join(
            '/data/users/user6/rxh/datasets/casme^2/74.220.215.205/Cropped224_all',
            subject, clip_name
        )

        frames = sorted([
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir) if f.endswith('.jpg')
        ])

        if len(frames) < L:
            continue

        n_frames = len(frames)
        window_iter = tqdm(range(0, n_frames - L + 1, S), 
                  desc=f"Process video frame window", 
                  total=(n_frames - L) // S + 1)

        for start in window_iter:
            end = start + L
            win_start, win_end = start, end

            init_frame = cv2.imread(frames[0], cv2.IMREAD_GRAYSCALE)
            local_flows, cum_flows = [], []
            prev_frame = None

            frame_iter = tqdm(range(start, end), 
                     desc=f"Compute optical flow [{start}-{end}]", 
                     total=L, 
                     leave=False)

            for t in frame_iter:
                curr_frame = cv2.imread(frames[t], cv2.IMREAD_GRAYSCALE)
                if prev_frame is None:
                    local_flows.append(0.0)
                else:
                    local_flows.append(compute_mag(prev_frame, curr_frame))
                cum_flows.append(compute_mag(init_frame, curr_frame))
                prev_frame = curr_frame

            local_flows = np.array(local_flows)
            cum_flows = np.array(cum_flows)
            combined = W1 * local_flows + W2 * cum_flows
            eps = 1e-6
            combined_norm = (combined - combined.min()) / (combined.max() - combined.min() + eps)

            covers_apex = apex >= win_start and apex < win_end
            overlap_len = max(0, min(win_end, offset) - max(win_start, onset))

            on_win, off_win = get_on_off_win(onset, offset, win_start, win_end, L)

            if covers_apex:
                ap_win = apex - win_start
                dhg_curve = gen_dhg(L, on_win, ap_win, off_win, BASE, PEAK)
                has_exp = True
            elif overlap_len >= L // 2 and USE_PSEUDO:
                local_peak = np.argmax(combined_norm[on_win:off_win+1]) + on_win
                ap_win = local_peak
                dhg_curve = gen_dhg(L, on_win, ap_win, off_win, BASE, PEAK)
                has_exp = True
            else:
                dhg_curve = np.zeros(L)
                ap_win = -1
                on_win = -1
                off_win = -1
                has_exp = False

            np.savez(
                os.path.join(output_dir, f"{subject}_{clip_name}_win{start}.npz"),
                local_flow=local_flows.astype(np.float32),
                cum_flow=cum_flows.astype(np.float32),
                combined_flow=combined_norm.astype(np.float32),
                micro_dhg=dhg_curve if expr_type == 'micro' else np.zeros(L),
                macro_dhg=dhg_curve if expr_type == 'macro' else np.zeros(L),
                meta=dict(
                    subject=subject,
                    clip=clip_name,
                    expr_type=expr_type,
                    win_start=win_start,
                    win_end=win_end,
                    onset=onset,
                    apex=apex,
                    offset=offset,
                    on_win=int(on_win),
                    ap_win=int(ap_win),
                    off_win=int(off_win),
                    has_exp = has_exp
                )
            )

if __name__ == "__main__":
    main()