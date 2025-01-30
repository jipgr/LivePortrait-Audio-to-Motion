from pathlib import Path
import imageio
from rich.progress import track
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import tyro

from src.a2m.config import Config
from src.a2m.dataset import AudioToMotionDataset,REGION_TO_VERTICES
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


def main(cfg:Config):
    if cfg.load_ckpt is None:
        log("Specify a checkpoint")
        exit(1)

    log(f"Load data from {cfg.data_root}")
    data = AudioToMotionDataset(
        cfg,
        split='all',
        timestep_start=120,
        timestep_end=180,
    )
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

    model.load_state_dict(torch.load(
        cfg.load_ckpt,
        map_location=cfg.device
    ))
    save_to = cfg.load_ckpt.parent / ( cfg.load_ckpt.stem+'.mp4' )

    model.eval()

    log("Setting up output ...")
    video_writer = None
    if save_to.suffix == '.mp4':
        video_writer = get_video_writer(str(save_to))
        log(f"Rendering to video {save_to} ...")
    elif not save_to.exists() or save_to.is_dir():
        save_to.mkdir(parents=True, exist_ok=True)
        log(f"Rendering to video {save_to} ...")
    else:
        raise ValueError(f"Dont know how to save to {save_to}")

    # smooth_fac = .5
    # prev_out = None

    exp_ids = data.expr_ids

    frame_i = 0
    for aud_feat, motion_gt in track(loader):
        aud_feat:torch.Tensor = aud_feat.to(cfg.device)
        motion_gt:torch.Tensor = motion_gt.to(cfg.device)

        exp_gt = torch.zeros(motion_gt.shape[0], 21, 3, dtype=motion_gt.dtype, device=motion_gt.device)
        exp_gt = torch.zeros(motion_gt.shape[0], 21, 3, dtype=motion_gt.dtype, device=motion_gt.device)
        exp_pred = torch.zeros(motion_gt.shape[0], 21, 3, dtype=motion_gt.dtype, device=motion_gt.device)


        with torch.no_grad():
            motion_pred = model(aud_feat)

        exp_gt[:,exp_ids] = motion_gt
        exp_pred[:, exp_ids] = motion_pred

        for item_i in range(motion_gt.shape[0]):
            render_gt_full = lp_pipe.render(motion_gt[item_i])

            # if prev_out is not None:
            #     pred_smooth = prev_out * smooth_fac + motion_pred[item_i] * (1. - smooth_fac)
            # else:
            #     pred_smooth = motion_pred[item_i]
            # prev_out = motion_pred[item_i]
            # render_pred = lp_pipe.render(pred_smooth)

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

    log("Done.")

if __name__ == '__main__':
    cfg = tyro.cli(Config)
    tyro.extras.set_accent_color("bright_cyan")
    main(cfg)
