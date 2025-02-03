import pickle
from typing import Literal
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset as _Dataset, DataLoader

from .config import Config,REGION_TO_VERTICES
from .utils import extract_wav
from .audioencoders import encode_audio
from src.utils.rprint import rprint as print,rlog as log

class AudioToMotionDataset(_Dataset):
    def __init__(
        self,
        cfg:Config,
        split:Literal['train','val','all'],
    ):
        super().__init__()
        self.cfg = cfg
        self.split = split

        if not cfg.data_root.exists():
            raise ValueError(f"Cannot find dataset at {cfg.data_root}")
        log(f"Loading dataset from {cfg.data_root}")

        self._load_motion()
        self._load_audio()

        num_frames = self.aud_features.shape[0]

        if self.motion is not None:
            assert num_frames == self.motion.shape[0], [self.aud_features.shape, self.motion.shape]
            log(f"Dataset motion features shape={self.motion.shape}")
            self.has_gt = True

        else:
            log(f"Dataset has no ground truth motion")
            self.has_gt = False

        # Indices used in this dataset
        indices = np.arange(num_frames)
        if self.split == 'train':
            N = int(num_frames * .2)
            assert 0 < N < num_frames - 1
            self._indices = indices[N:]

        elif self.split == 'val':
            N = int(num_frames * .2)
            assert 0 < N < num_frames - 1
            self._indices = indices[:N]

        else:
            self._indices = indices

        log(f"Full dataset has {num_frames} frames")
        log(f"Data split '{self.split}' has {len(self)} frames")
        log(f"Data FPS: {self.fps}")

    def _load_motion(self):

        path_pkl = self.cfg.data_root / f'{self.cfg.data_root.name}.pkl'
        if path_pkl.is_file():

            self.expr_ids = REGION_TO_VERTICES.get(self.cfg.region,[])
            assert self.expr_ids != [], self.cfg.region

            with open(path_pkl, 'rb') as f:
                template = pickle.load(f)

                motion = np.concatenate([motion['exp'] for motion in template['motion']])
                fps:float = template['output_fps']

            self.fps = fps
            self.motion = motion
            self.m0 = motion[0]

        else:
            log(f"No motion file found at {path_pkl}, no ground truth available")
            self.fps = 25.
            self.motion = None

    def _load_audio(self) -> np.ndarray:
        # Either pointing to 'aud_[encoder].npy' or '*.[wav|mp4]'
        if self.cfg.data_root.is_file():

            if self.cfg.data_root.suffix == '.npy':
                path_feats = self.cfg.data_root
                path_aud = None

            elif self.cfg.data_root.suffix in {'.wav', '.mp4'}:
                path_feats = self.cfg.data_root.parent / f'aud_{self.cfg.audio_encoder}.npy'
                path_aud = self.cfg.data_root

            else:
                raise ValueError(f"Expected datafile to have ext .npy,.wav,.mp4")

        # Data root is dir
        else:
            path_feats = self.cfg.data_root / f'aud_{self.cfg.audio_encoder}.npy'

            # Features exist, no need to encode audio
            if path_feats.is_file():
                path_aud = None

            # Extract from aud.wav
            elif ( self.cfg.data_root / 'aud.wav' ).is_file():
                path_aud = self.cfg.data_root / 'aud.wav'

            # Extract from .../[name]/[name].mp4
            elif ( self.cfg.data_root / f'{self.cfg.data_root.name}.mp4' ).is_file():
                path_aud = self.cfg.data_root / f'{self.cfg.data_root.name}.mp4'

            # Now I also dont know what to do any more
            else:
                raise ValueError(f"Expected either 'aud.wav' or '*.mp4' in {self.cfg.data_root}")

        encoder_kwargs = {
            'fps': self.fps
        }
        if self.cfg.audio_encoder == 'synctalk':
            encoder_kwargs['enc_ckpt'] = self.cfg.audio_enc_ckpt_synctalk

        self.aud_features = encode_audio(
            path_aud=path_aud,
            path_feats=path_feats,
            encoder=self.cfg.audio_encoder,
            num_frames=None if self.motion is None else self.motion.shape[0],
            encoder_kwargs=encoder_kwargs,
        )

        assert self.aud_features.shape[-1] == self.cfg.dim_aud, (self.aud_features.shape, self.cfg.dim_aud)

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

        if self.motion is not None:
            motion = self.motion[feat_idx] - self.m0
            expr = torch.from_numpy( motion[self.expr_ids] )
        else:
            expr = []

        return aud, expr
