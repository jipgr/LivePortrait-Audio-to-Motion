from pathlib import Path
import imageio
import rich
from rich.progress import track
import numpy as np
import rich.progress
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import tyro

from src.a2m.config import InferenceConfig
from src.a2m.dataset import AudioToMotionDataset,REGION_TO_VERTICES
from src.a2m.model import AudioToMotionModel
from src.utils.rprint import rprint as print,rlog as log
from src.live_portrait_pipeline_a2m import LPPipeA2M


def get_video_writer(wfp, **kwargs):
    fps = kwargs.get('fps', 25)
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



def main(cfg:InferenceConfig):
    if cfg.load_ckpt is None:
        raise ValueError("Specify a checkpoint to load")

    log(f"Loading data ....")
    data = AudioToMotionDataset(
        cfg,
        split=cfg.render_split,
    )
    loader = DataLoader(
        dataset=data,
        batch_size=int(data.fps),
        shuffle=False,
        num_workers=cfg.num_worker
    )

    log("Loading LivePortrait models ...")
    lp_pipe = LPPipeA2M(
        source='data/obama/obama.png',
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
    save_to = cfg.load_ckpt.parent / f'{cfg.load_ckpt.stem}_{cfg.data_root.stem}.mp4'

    model.eval()

    log("Setting up output ...")
    video_writer = None
    if save_to.suffix == '.mp4':
        video_writer = get_video_writer(
            str(save_to),
            fps=data.fps
        )
        log(f"Rendering to video {save_to} ...")
    elif not save_to.exists() or save_to.is_dir():
        save_to.mkdir(parents=True, exist_ok=True)
        log(f"Rendering to video {save_to} ...")
    else:
        raise ValueError(f"Dont know how to save to {save_to}")

    # smooth_fac = .5
    # prev_out = None

    frame_i = 0
    for aud_feat, motion_gt in track(loader):

        aud_feat:torch.Tensor = aud_feat.to(cfg.device)

        if data.has_gt:
            motion_gt:torch.Tensor = motion_gt.to(cfg.device)

        with torch.no_grad():
            motion_pred:torch.Tensor = model(aud_feat)

        for item_i in range(motion_pred.shape[0]):

            # if prev_out is not None:
            #     pred_smooth = prev_out * smooth_fac + motion_pred[item_i] * (1. - smooth_fac)
            # else:
            #     pred_smooth = motion_pred[item_i]
            # prev_out = motion_pred[item_i]
            # render_pred = lp_pipe.render(pred_smooth)


            renders = lp_pipe.render(motion_pred[item_i])

            if data.has_gt:
                render_gt = lp_pipe.render(motion_gt[item_i])
                renders = torch.cat((render_gt, renders), dim=-1)

            if video_writer is not None:
                renders_np = np.uint8(255*renders.squeeze(0).permute(1,2,0).detach().cpu().numpy())
                video_writer.append_data(renders_np)
            else:
                save_image(renders, save_to / f'frame{frame_i:06d}.png')

            frame_i += 1

        if frame_i > data.fps*10:
            break

    if video_writer is not None:
        video_writer.close()

    log("Done.")

if __name__ == '__main__':
    cfg = tyro.cli(InferenceConfig)
    tyro.extras.set_accent_color("bright_cyan")
    main(cfg)
