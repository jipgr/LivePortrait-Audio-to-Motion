from pathlib import Path
import torch
import torch.nn.functional as F
import torchaudio

"""
For a video of length T second, audio at sample rate 16KHz has T*16000 samples
HuBERT takes 320 samples at a time, so that gives T*16000/320 = T*50 features
by default.

For a given fps (features per second, same as the video fps), F, we actually
want T*F frames
"""

def encode_audio(
    path_in:Path,
    path_out:Path=None,
    fps:float=25.
):

    if path_out is None:
        path_out = path_in.with_name('aud_hubert.npy')

    torch.random.manual_seed(0)
    device = torch.device("cuda")

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H
    print("Model sample Rate:", bundle.sample_rate)

    model:torchaudio.models.Wav2Vec2Model = bundle.get_model().to(device)

    waveform, input_sr = torchaudio.load(path_in)
    T = waveform.shape[1] / input_sr # length of the audio in seconds
    num_frames = round(T*fps) # number of output features/frames

    print(f"Waveform has {waveform.shape[1]} samples, gives a length of {T} seconds")

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    if waveform.shape[1] > 1:
        print("!WARNING! multiple audio channels as input, only using the first")
    waveform = waveform[:1,:]

    waveform = waveform.to(device)

    if input_sr != bundle.sample_rate:
        print(f"!WARNING! Resampling input audio ({input_sr}Hz) to model sample rate ({bundle.sample_rate}Hz)")
        waveform = torchaudio.functional.resample(waveform, input_sr, bundle.sample_rate)


    with torch.inference_mode():
        features, _ = model.extract_features(waveform)
        # Only take the last layers output
        features = features[-1]

    # HuBERT has a stride of 320 audio samples, at 16KHz this is 50FPS
    features = features.permute(0,2,1) # batchsize, num_features, num_samples = 1, 1024, T*50

    # Resample this to the target FPS
    features:torch.Tensor = F.interpolate(features, size=num_frames)

    # Send out as np array
    features = features.squeeze(0).t().detach().cpu().numpy() # num_frames, num_features
    return features
