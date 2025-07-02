import hydra
from omegaconf import DictConfig
import os
import re
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial

NUM_RE = re.compile(r'(\d+)')
def extract_frame_number(filename):
    basename = os.path.basename(filename)
    match = re.search(r'img(\d+)\.jpg', basename)
    return int(match.group(1))

def gen_dhg(L, on_win, ap_win, off_win, BASE, PEAK):
    sigma_l = max((ap_win - on_win) / 3.0, 1.0)
    sigma_r = max((off_win - ap_win) / 3.0, 1.0)
    curve = np.zeros(L)
    for i in range(on_win, off_win + 1):
        delta = (i - ap_win) / sigma_l if i <= ap_win else (i - ap_win) / sigma_r
        curve[i] = BASE + (PEAK - BASE) * np.exp(-0.5 * delta**2)
    curve[0] = 0.0; curve[on_win] = BASE; curve[ap_win] = PEAK; curve[off_win] = BASE; curve[-1] = 0.0
    if on_win > 0:
        curve[:on_win] = np.linspace(0.0, BASE, on_win, endpoint=False)
    if off_win + 1 < L:
        curve[off_win + 1:] = np.linspace(BASE, 0.0, L - (off_win + 1))
    return curve

def get_on_off_win(onset, offset, win_start, win_end, L):
    on_win = max(onset - win_start, 0) if onset >= win_start else 0
    off_win = L - 1 if offset >= win_end else max(min(offset, win_end) - win_start, 0)
    return on_win, off_win

def process_video(cfg, video_name, group):
    L = cfg.preprocess.window_size
    S = cfg.preprocess.stride
    BASE = cfg.preprocess.dhg.base
    PEAK = cfg.preprocess.dhg.peak
    USE_PSEUDO = cfg.preprocess.pseudo_apex.use_if_overlap_half

    first_row = group.iloc[0]
    sub_id = int(first_row['subject'])
    subject = f"s{sub_id:02d}"

    frames_dir = Path(cfg.preprocess.input_dir) / subject / video_name
    curve_path = Path(cfg.preprocess.flow_curve_dir) / subject / video_name / 'curve.npy'
    output_dir = Path(cfg.preprocess.output_dir) / f"{subject}_{video_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    curve = np.load(curve_path)
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')], key=extract_frame_number)
    if len(frames) < L:
        return

    n_frames = len(frames)
    for start in tqdm(range(0, n_frames - L + 1, S), desc=f"{subject}/{video_name}"):
        end = start + L
        win_curve = curve[start:end]

        matches = []
        for _, row in group.iterrows():
            onset, offset = int(row['start_frame']), int(row['end_frame'])
            iou = max(0, min(end, offset) - max(start, onset)) / (max(end, offset) - min(start, onset) + 1e-6)
            matches.append((iou, row))

        micro_matches = [(iou, row) for iou, row in matches if int(row['type_idx']) == 0]
        macro_matches = [(iou, row) for iou, row in matches if int(row['type_idx']) == 1]

        micro_dhg = np.zeros(L)
        macro_dhg = np.zeros(L)

        if micro_matches:
            max_micro = max(micro_matches, key=lambda x: x[0])
            iou, row = max_micro
            onset, apex, offset = int(row['start_frame']), int(row['apex_frame']), int(row['end_frame'])
            covers_apex = apex >= start and apex < end
            overlap_len = max(0, min(end, offset) - max(start, onset))
            on_win, off_win = get_on_off_win(onset, offset, start, end, L)
            if covers_apex:
                ap_win = apex - start
                micro_dhg = gen_dhg(L, on_win, ap_win, off_win, BASE, PEAK)
            elif overlap_len >= L // 2 and USE_PSEUDO:
                local_peak = np.argmax(win_curve[on_win:off_win+1]) + on_win
                ap_win = local_peak
                micro_dhg = gen_dhg(L, on_win, ap_win, off_win, BASE, PEAK)

        if macro_matches:
            max_macro = max(macro_matches, key=lambda x: x[0])
            iou, row = max_macro
            onset, apex, offset = int(row['start_frame']), int(row['apex_frame']), int(row['end_frame'])
            covers_apex = apex >= start and apex < end
            overlap_len = max(0, min(end, offset) - max(start, onset))
            on_win, off_win = get_on_off_win(onset, offset, start, end, L)
            if covers_apex:
                ap_win = apex - start
                macro_dhg = gen_dhg(L, on_win, ap_win, off_win, BASE, PEAK)
            elif overlap_len >= L // 2 and USE_PSEUDO:
                local_peak = np.argmax(win_curve[on_win:off_win+1]) + on_win
                ap_win = local_peak
                macro_dhg = gen_dhg(L, on_win, ap_win, off_win, BASE, PEAK)

        np.savez(
            output_dir / f"win{start}.npz",
            flow_curve = win_curve.astype(np.float32),
            micro_dhg  = micro_dhg.astype(np.float32),
            macro_dhg  = macro_dhg.astype(np.float32),
            meta = dict(
                subject = subject,
                video   = video_name,
                win_start = start,
                win_end   = end
            )
        )

@hydra.main(config_path="/data/users/user6/hmr/code/microspot/configs", config_name="preprocess.yaml")
def main(cfg: DictConfig):
    anno_df = pd.read_csv(cfg.paths.anno_file)
    video_groups = anno_df.groupby('video_name')

    with ProcessPoolExecutor(max_workers=cfg.preprocess.num_workers) as executor:
        func = partial(process_video, cfg)
        list(tqdm(executor.map(func, video_groups.groups.keys(),
                               [video_groups.get_group(name) for name in video_groups.groups.keys()]),
                  total=len(video_groups), desc='Processing videos'))

if __name__ == "__main__":
    main()