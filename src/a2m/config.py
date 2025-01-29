import dataclasses
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
assert PROJECT_DIR.exists(), "I dont know on what device Im running :("

@dataclasses.dataclass
class Config:
    data_root:Path=PROJECT_DIR / 'data/theo'

    dim_aud:int=512
    region:Literal["exp", "pose", "lip", "eyes", "all"]='lip'

    model_depth:int=16
    model_width:int=128

    lr:float=1e-4
    num_epoch:int=500
    batch_size:int=128
    num_worker:int=2

    ckpt_every:int=25
    eval_every:int=25

    device:torch.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    load_ckpt:None|Path=None
    model_dir:Path= PROJECT_DIR / 'models' / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    audio_enc_ckpt:Path=PROJECT_DIR / 'pretrained_weights/synctalk/audio_visual_encoder.pth'

    def __post_init__(self):
        self.num_verts = len(REGION_TO_VERTICES.get(self.region, []))
