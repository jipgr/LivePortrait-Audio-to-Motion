import pickle
from typing import Literal
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset as _Dataset, DataLoader

from .config import Config,REGION_TO_VERTICES
from .audio import encode_audio

TDataItem = tuple[torch.Tensor, torch.Tensor]
TDataBatch = tuple[torch.Tensor, torch.Tensor]

class AudioToMotionDataset(_Dataset):
    def __init__(
        self,
        cfg:Config,
        split:Literal['train','val','all'],
    ):
        super().__init__()
        self.cfg = cfg

        self.split = split

        self.expr_ids = REGION_TO_VERTICES.get(cfg.region,[])
        assert self.expr_ids != [], cfg.region

        with open(cfg.data_root / 'motion.pkl', 'rb') as f:
            template = pickle.load(f)

            motion = np.concatenate([motion['exp'] for motion in template['motion']])
            self.fps = template['output_fps']

        self.motion = motion

        if (cfg.data_root / 'aud.npy').is_file():
            self.aud_features = np.load(cfg.data_root / 'aud.npy')
        else:
            self.aud_features = encode_audio(
                path_wav=cfg.data_root / 'aud.wav',
                fps=self.fps,
                enc_ckpt=cfg.audio_enc_ckpt,
            )
            np.save(cfg.data_root / 'aud.npy', self.aud_features)
        assert self.aud_features.shape[-1] == cfg.dim_aud


        num_frames = self.aud_features.shape[0]

        assert num_frames == self.motion.shape[0], [self.aud_features.shape, self.motion.shape]

        # print("!WARNING! using a fixed max 600 frames")
        # num_frames = 600
        # self.motion = self.motion[:num_frames]
        # self.aud_features = self.aud_features[:num_frames]

        if self.split == 'train':
            N = int(num_frames * .8)
            assert 0 < N < num_frames - 1

            self._indices = np.arange(N)

        elif self.split == 'val':
            N = int(num_frames * .8)
            assert 0 < N < num_frames - 1

            self._indices = np.arange(N, num_frames)

        else:
            self._indices = np.arange(num_frames)

    def dataloader(self):
        return DataLoader(
            dataset=self,
            batch_size=self.cfg.batch_size if self.split == 'val' else 1,
            shuffle=True if self.split == 'train' else False,
            num_workers=self.cfg.num_worker
        )

    def __len__(self):
        return self._indices.size


    def __getitem__(self, data_idx):
        feat_idx = self._indices[data_idx]
        aud = torch.from_numpy( self.aud_features[feat_idx] )
        motion = self.motion[feat_idx] - self.motion[0]
        expr = torch.from_numpy( motion[self.expr_ids] )

        return aud, expr
