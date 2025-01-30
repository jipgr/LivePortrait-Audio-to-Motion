import pickle
from typing import Literal
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset as _Dataset, DataLoader

from .config import Config,REGION_TO_VERTICES
from .utils import extract_wav
from src.utils.rprint import rprint as print,rlog as log

TDataItem = tuple[torch.Tensor, torch.Tensor]
TDataBatch = tuple[torch.Tensor, torch.Tensor]


class AudioToMotionDataset(_Dataset):
    def __init__(
        self,
        cfg:Config,
        split:Literal['train','val','all'],
        timestep_start:float=0,
        timestep_end:float|None=None,
    ):
        super().__init__()
        self.cfg = cfg

        self.split = split

        self.expr_ids = REGION_TO_VERTICES.get(cfg.region,[])
        assert self.expr_ids != [], cfg.region

        with open(cfg.data_root / 'motion.pkl', 'rb') as f:
            template = pickle.load(f)

            motion = np.concatenate([motion['exp'] for motion in template['motion']])
            self.fps:float = template['output_fps']

        log(f"Data FPS: {self.fps}")

        self.motion = motion
        self.m0 = motion[0]

        self.aud_features = self._encode_audio()
        assert self.aud_features.shape[-1] == cfg.dim_aud

        num_frames = self.aud_features.shape[0]
        assert num_frames == self.motion.shape[0], [self.aud_features.shape, self.motion.shape]
        log(f"Full dataset has {num_frames} frames")

        if timestep_start != 0 or timestep_end is not None:
            frame_start = max(0, round(timestep_start * self.fps))
            frame_end = min(num_frames, round(timestep_end * self.fps) if timestep_end is not None else num_frames)
            num_frames = frame_end - frame_start
            log(f"Using data slice {timestep_start}s:{timestep_end}s -> frames {frame_start}:{frame_end}")
        else:
            frame_start,frame_end = 0,num_frames

        # print("!WARNING! using a fixed max 600 frames")
        # num_frames = 600
        # self.motion = self.motion[:num_frames]
        # self.aud_features = self.aud_features[:num_frames]

        if self.split == 'train':
            N = int(num_frames * .8)
            assert 0 < N < num_frames - 1

            self._indices = frame_start + np.arange(N)

        elif self.split == 'val':
            N = int(num_frames * .8)
            assert 0 < N < num_frames - 1

            self._indices = frame_start + np.arange(N, num_frames)

        else:
            self._indices = frame_start + np.arange(num_frames)

        log(f"Data split '{self.split}' has {len(self)} frames")


    def _encode_audio(self) -> np.ndarray:
        path_wav = self.cfg.data_root / 'aud.wav'

        if not path_wav.is_file():
            extract_wav(self.cfg.data_root / (self.cfg.data_root.name + '.mp4'), path_wav)

        save_to = self.cfg.data_root / f'aud_{self.cfg.audio_encoder}.npy'

        if save_to.is_file():
            log(f"Loading audio features from {save_to}")
            features = np.load(save_to)

        else:
            if self.cfg.audio_encoder == 'synctalk':
                from .audioencoders.synctalk import encode_audio
                features = encode_audio(path_wav, fps=self.fps, enc_ckpt=self.cfg.audio_enc_ckpt_synctalk)

            elif self.cfg.audio_encoder == 'hubert':
                from .audioencoders.hubert import encode_audio
                features = encode_audio(path_wav, fps=self.fps)

            else:
                raise ValueError(f"Unkown audio encoder {self.cfg.audio_encoder}")

            assert features.shape == (self.motion.shape[0], self.cfg.dim_aud), \
                "Audio features are incorrect, expected shape {}, got {}".format(
                    (self.motion.shape[0], self.cfg.dim_aud), features.shape)

            log(f"Saving features to {save_to}")
            np.save(save_to, features)

        return features

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

        motion = self.motion[feat_idx] - self.m0
        expr = torch.from_numpy( motion[self.expr_ids] )

        return aud, expr
