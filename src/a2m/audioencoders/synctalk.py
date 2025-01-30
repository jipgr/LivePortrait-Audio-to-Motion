import numpy as np
import torch

from ..utils import load_wav, melspectrogram
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, leakyReLU=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        if leakyReLU:
            self.act = nn.LeakyReLU(0.02)
        else:
            self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        out:torch.Tensor = self.audio_encoder(x)
        out = out.squeeze(2).squeeze(2)

        return out


class AudioDataset(Dataset):
    def __init__(self, wavpath, fps):
        super().__init__()

        self.fps = fps

        wav = load_wav(wavpath, 16000)
        self.orig_mel = melspectrogram(wav).T
        print(f"Wav shape: {wav.shape}")
        print(f"Mel shape: {self.orig_mel.shape}")

        self.data_len = round((self.orig_mel.shape[0] - 16) / 80. * float(self.fps) + .5) + 2

    def crop_audio_window(self, spec, start_frame):
        start_idx = int(80. * (start_frame / float(self.fps)))

        end_idx = start_idx + 16
        if end_idx > spec.shape[0]:
            # print(end_idx, spec.shape[0])
            end_idx = spec.shape[0]
            start_idx = end_idx - 16

        return spec[start_idx: end_idx, :]

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):

        mel = self.crop_audio_window(self.orig_mel.copy(), idx)
        if (mel.shape[0] != 16):
            raise Exception('mel.shape[0] != 16')
        mel = torch.FloatTensor(mel.T).unsqueeze(0)

        return mel

def encode_audio(
    path_wav:str,
    fps:float,
    enc_ckpt:str,
) -> np.ndarray:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AudioEncoder().to(device).eval()
    model.load_state_dict({f'audio_encoder.{k}': v for k, v in torch.load(enc_ckpt).items()})

    dataset = AudioDataset(path_wav, fps=fps)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

    outputs = []

    for mel in data_loader:
        mel = mel.to(device)

        with torch.no_grad():
            out = model(mel)

        outputs.append(out)

    outputs = torch.cat(outputs, dim=0).cpu()
    first_frame, last_frame = outputs[:1], outputs[-1:]
    features = torch.cat([first_frame.repeat(2, 1), outputs, last_frame.repeat(2, 1)], dim=0).numpy()

    return features
