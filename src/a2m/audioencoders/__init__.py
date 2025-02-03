import os
from pathlib import Path
from typing import Literal

import numpy as np
import scipy.interpolate

from .synctalk import encode_audio as encode_audio_synctalk
from .hubert import encode_audio as encode_audio_hubert

def extract_wav(path_vid, path_wav, sr=16000):
    cmd = f'ffmpeg -y -loglevel error -i {path_vid} -f wav -ar {sr} {path_wav}'
    os.system(cmd)

METHODS = {
    'synctalk': encode_audio_synctalk,
    'hubert': encode_audio_hubert
}

def encode_audio(
    path_aud:Path|None,
    path_feats:Path,
    encoder:Literal['synctalk','hubert'],
    num_frames:int|None=None,
    encoder_kwargs:dict={}
):
    if path_feats.is_file():
        print(f"Loading audio features from {path_feats} ...")
        features = np.load(path_feats)

    elif path_aud is None:
        raise ValueError(f"Features {path_feats} not found, but no audio supplied")

    elif encoder in METHODS:
        print(f"Extracting features using {encoder} ...")
        features:np.ndarray = METHODS[encoder](path_aud, **encoder_kwargs)

        print(f"Extracted features shape:", features.shape)

        if num_frames is not None and features.shape[0] != num_frames:
            print(f"!WARNING! Extracted features are of shape {features.shape}, but expected {num_frames} frames")
            print("Features are resampled using linear torch interpolate")
            import torch
            from torch.nn.functional import interpolate
            features_ten = torch.from_numpy(features).t().unsqueeze(0) # 1, feature_dim, features.shape[0]
            features_ten = interpolate(features_ten, num_frames, mode='linear')
            features = features_ten.squeeze(0).t().numpy()

            # if features.shape[0] == num_frames - 1:
            #     print(f"Audio features has one frame less than expected, copying the latest audio feature to match expected length")
            #     features = np.concatenate((features, features[-1:]), axis=0)
            # elif features.shape[0] == num_frames + 1:
            #     print(f"Audio features has one frame more than expected, dropping last audio features to match expected length")
            #     features = features[:-1]

            # elif features.shape[0] != num_frames:
            #     raise ValueError(f"Expected {num_frames} audio features, got shape {features.shape}")

        print(f"Saving features to {path_feats}")
        np.save(path_feats, features)

    else:
        raise ValueError(f"Unkown audio encoder {encoder}")

    return features
