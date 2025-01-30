from dataclasses import dataclass
import datetime
from pathlib import Path
from typing import Literal

import torch

REGION_TO_VERTICES = {
    'all': list(range(21)),
    'exp': list(range(21)),
    'lip': [6, 12, 14, 17, 19, 20],
    'eyes': [11, 13, 15, 16, 18],
}

PROJECT_DIR = Path('/app/' if Path('/app/').exists() else '/home/jip/data2/LivePortrait/')
assert PROJECT_DIR.exists(), "I dont know on what host Im running on :("


@dataclass
class Config:
    data_root:Path

    audio_encoder:Literal['synctalk', 'hubert']

    region:Literal["exp", "pose", "lip", "eyes", "all"]='lip'

    model_depth:int=16
    model_width:int=128

    batch_size:int=1
    num_worker:int=2

    lr:float=1e-4
    num_epoch:int=500
    batch_size:int=128

    ckpt_every:int=25
    eval_every:int=25
    model_dir:Path= PROJECT_DIR / 'models' / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    load_ckpt:Path|None=None

    device:torch.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    audio_enc_ckpt_synctalk:Path=PROJECT_DIR / 'pretrained_weights/synctalk/audio_visual_encoder.pth'

    def __post_init__(self):
        self.num_verts = len(REGION_TO_VERTICES.get(self.region, []))

        self.dim_aud = {
            'synctalk': 512,
            'hubert': 1024,
        }[self.audio_encoder]
