from pathlib import Path
import imageio
from rich.progress import track
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import tyro

from src.a2m.config import Config
from src.a2m.dataset import AudioToMotionDataset
from src.a2m.model import AudioToMotionModel
from src.utils.rprint import rprint as print,rlog as log
from src.live_portrait_pipeline_a2m import LPPipeA2M

def get_video_writer(wfp, **kwargs):
    fps = kwargs.get('fps', 30)
    video_format = kwargs.get('format', 'mp4')  # default is mp4 format
    codec = kwargs.get('codec', 'libx264')  # default is libx264 encoding
    quality = kwargs.get('quality')  # video quality
    pixelformat = kwargs.get('pixelformat', 'yuv420p')  # video pixel format
    macro_block_size = kwargs.get('macro_block_size', 2)
    ffmpeg_params = ['-crf', str(kwargs.get('crf', 18))]

    return imageio.get_writer(
        wfp, fps=fps, format=video_format,
        codec=codec, quality=quality, ffmpeg_params=ffmpeg_params, pixelformat=pixelformat, macro_block_size=macro_block_size
    )

def render(
    model:AudioToMotionModel,
    lp_pipe:LPPipeA2M,
    loader:DataLoader,
    save_to:Path,
    device:torch.device,
):
    model.eval()

    video_writer = None
    if save_to.suffix == '.mp4':
        video_writer = get_video_writer(str(save_to))
        log(f"Rendering to video {save_to} ...")
    elif not save_to.exists() or save_to.is_dir():
        save_to.mkdir(parents=True, exist_ok=True)
        log(f"Rendering to video {save_to} ...")
    else:
        raise ValueError(f"Dont know how to save to {save_to}")

    frame_i = 0
    for aud_feat, motion_gt in track(loader):
        aud_feat:torch.Tensor = aud_feat.to(device)
        motion_gt:torch.Tensor = motion_gt.to(device)

        with torch.no_grad():
            motion_pred = model(aud_feat)

        for item_i in range(motion_gt.shape[0]):
            render_gt = lp_pipe.render(motion_gt[item_i])
            render_pred = lp_pipe.render(motion_pred[item_i])

            renders = torch.cat((render_gt, render_pred), dim=-1)

            if video_writer is not None:
                renders_np = np.uint8(255*renders.squeeze(0).permute(1,2,0).detach().cpu().numpy())
                video_writer.append_data(renders_np)
            else:
                save_image(renders, save_to / f'frame{frame_i:06d}.png')

            frame_i += 1

    if video_writer is not None:
        video_writer.close()


def main(cfg:Config):
    log(f"Load data from {cfg.data_root}")
    data = AudioToMotionDataset(cfg, 'all')
    loader = DataLoader(
        dataset=data,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_worker
    )

    log("Loading LivePortrait models ...")
    lp_pipe = LPPipeA2M(
        source='assets/examples/source/theo.png',
        animation_region=cfg.region,
        device_id=cfg.device.index,
    )

    log("Setting up Audio-to-Motion model ...")
    model = AudioToMotionModel(cfg)

    model.to(cfg.device)

    if cfg.load_ckpt is not None:
        model.load_state_dict(torch.load(
            cfg.load_ckpt,
            map_location=cfg.device
        ))
        cfg.model_dir = cfg.load_ckpt.parent
        save_to = cfg.load_ckpt.parent / ( cfg.load_ckpt.stem+'.mp4' )
    else:
        log("!WARNING! No checkpoint provided, running inference on a random model")
        save_to = cfg.model_dir / f'render.mp4'

    render(
        model=model,
        lp_pipe=lp_pipe,
        loader=loader,
        device=cfg.device,
        save_to=save_to,
    )

    log("Done.")

if __name__ == '__main__':
    cfg = tyro.cli(Config)
    tyro.extras.set_accent_color("bright_cyan")
    main(cfg)
