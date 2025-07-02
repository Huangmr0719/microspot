import hydra
from omegaconf import DictConfig
import os
import re
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig
import torch.nn as nn

def extract_frame_number(filename):
    basename = os.path.basename(filename)
    match = re.search(r'img(\d+)\.jpg', basename)
    return int(match.group(1))

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))
])

@hydra.main(config_path="/data/users/user6/hmr/code/microspot/configs", config_name="vis_extract.yaml")
def main(cfg: DictConfig):
    clip_len = 16
    stride = 16
    aggregate_mode = 'concat'  # mean or concat

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = AutoConfig.from_pretrained(cfg.model.name, trust_remote_code=True)
    processor = VideoMAEImageProcessor.from_pretrained(cfg.model.name)
    model = AutoModel.from_pretrained(cfg.model.name, config=config, trust_remote_code=True).to(device).eval()

    npz_files = list(Path(cfg.paths.preprocess_dir).rglob('*.npz'))
    print(f"Found {len(npz_files)} window files to process.")

    for npz_file in tqdm(npz_files, desc="Processing VideoMAE"):
        output_path = npz_file.parent / f"{npz_file.stem}_videomae.npz"
        if output_path.exists():
            continue

        data = np.load(npz_file, allow_pickle=True)
        meta = data['meta'].item()
        win_start = meta['win_start']
        win_end   = meta['win_end']

        subject = meta['subject']
        video   = meta['video']

        frames_dir = Path(cfg.paths.input_dir) / subject / video
        frame_files = sorted(frames_dir.glob('*.jpg'), key=extract_frame_number)
        clip_files = frame_files[win_start:win_end]

        frames = []
        for f in clip_files:
            img = cv2.imread(str(f))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)

        all_feats = []
        for start in range(0, len(frames) - clip_len + 1, stride):
            clip = frames[start : start + clip_len]

            # 如果不足 clip_len 就 padding 最后一帧
            if len(clip) < clip_len:
                clip += [clip[-1]] * (clip_len - len(clip))

            inputs = processor(clip, return_tensors="pt")
            inputs['pixel_values'] = inputs['pixel_values'].permute(0, 2, 1, 3, 4).to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                feat = outputs.mean(dim=1).cpu().numpy()  # [N,D]
                all_feats.append(feat.squeeze())

        all_feats = np.stack(all_feats, axis=0)  # [num_clips, D]

        if aggregate_mode == 'mean':
            final_feat = np.mean(all_feats, axis=0)
        elif aggregate_mode == 'concat':
            final_feat = all_feats.flatten()
        else:
            raise ValueError(f"Unknown aggregate mode: {aggregate_mode}")

        np.savez(
            output_path,
            **{k: data[k] for k in data.keys()},
            videomae_feature=final_feat.astype(np.float32)
        )

    print(f"✅ VideoMAE features saved to {cfg.paths.preprocess_dir}")

if __name__ == "__main__":
    main()