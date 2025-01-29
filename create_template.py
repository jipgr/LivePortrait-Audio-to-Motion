import os
import sys
import os.path as osp
import tyro
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from live_portrait_pipeline_a2m import LivePortraitPipeline
from inference import partial_fields, fast_check_args

import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import numpy as np
import os
import os.path as osp
from rich.progress import track

import torch
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.utils.camera import get_rotation_matrix
from src.utils.video import images2video, concat_frames, add_audio_to_video, has_audio_stream
from src.utils.crop import prepare_paste_back, paste_back
from src.utils.io import load_image_rgb, resize_to_limit, dump, load
from src.utils.helper import mkdir, basename, dct2device, is_template, is_image, calc_motion_multiplier
from src.utils.cropper import Cropper
from src.utils.camera import get_rotation_matrix
from src.utils.video import images2video, concat_frames, get_fps, add_audio_to_video, has_audio_stream
from src.utils.crop import prepare_paste_back, paste_back
from src.utils.io import load_image_rgb, load_video, resize_to_limit, dump, load
from src.utils.helper import mkdir, basename, dct2device, is_video, is_template, remove_suffix, is_image, is_square_video, calc_motion_multiplier
from src.utils.filter import smooth
from src.utils.rprint import rlog as log

def prepare_frame(img, device):
    """ construct the input as standard
    imgs: BxHxWx3, uint8
    """
    _img = np.array(img)[..., np.newaxis]  # HxWx3x1

    y = _img.astype(np.float32) / 255.
    y = np.clip(y, 0, 1)  # clip to 0~1
    y = torch.from_numpy(y).permute(3,2,0,1)  # HxWx3x1 -> 1x3xHxW
    y = y.to(device)

    return y

def make_template(pl:LivePortraitPipeline, driving_rgb_crop_256x256_lst, c_eyes_lst, c_lip_lst, **kwargs):
    n_frames = len(driving_rgb_crop_256x256_lst)
    template_dct = {
        'n_frames': n_frames,
        'output_fps': kwargs.get('output_fps', 25),
        'motion': [],
        'c_eyes_lst': [],
        'c_lip_lst': [],
    }

    for i in track(range(n_frames), description='Making motion templates...', total=n_frames):
        # collect s, R, δ and t for inference
        I_i = prepare_frame(driving_rgb_crop_256x256_lst[i], device=pl.live_portrait_wrapper.device)
        x_i_info = pl.live_portrait_wrapper.get_kp_info(I_i)
        x_s = pl.live_portrait_wrapper.transform_keypoint(x_i_info)
        R_i = get_rotation_matrix(x_i_info['pitch'], x_i_info['yaw'], x_i_info['roll'])

        item_dct = {
            'scale': x_i_info['scale'].cpu().numpy().astype(np.float32),
            'R': R_i.cpu().numpy().astype(np.float32),
            'exp': x_i_info['exp'].cpu().numpy().astype(np.float32),
            't': x_i_info['t'].cpu().numpy().astype(np.float32),
            'kp': x_i_info['kp'].cpu().numpy().astype(np.float32),
            'x_s': x_s.cpu().numpy().astype(np.float32),
        }

        template_dct['motion'].append(item_dct)

        c_eyes = c_eyes_lst[i].astype(np.float32)
        template_dct['c_eyes_lst'].append(c_eyes)

        c_lip = c_lip_lst[i].astype(np.float32)
        template_dct['c_lip_lst'].append(c_lip)

    return template_dct

log("Setting up args ...")
tyro.extras.set_accent_color("bright_cyan")
args = tyro.cli(ArgumentConfig, args=[
    '-s', 'assets/examples/source/theo.png',
    '-d', 'assets/examples/driving/theo1.mp4',
    # '-d', 'assets/examples/driving/d0.pkl',
    '--animation_region', 'lip'
])

fast_check_args(args)

# specify configs for inference
inference_cfg = partial_fields(InferenceConfig, args.__dict__)
crop_cfg = partial_fields(CropConfig, args.__dict__)

log("Setting up pipeline ...")
pl = LivePortraitPipeline(
    inference_cfg=inference_cfg,
    crop_cfg=crop_cfg,
)

log("Reading input ...")
output_fps = int(get_fps(args.driving))
driving_rgb_lst = load_video(args.driving)

log("Cropping video ...")
ret_d = pl.cropper.crop_driving_video(driving_rgb_lst)
driving_rgb_crop_lst, driving_lmk_crop_lst = ret_d['frame_crop_lst'], ret_d['lmk_crop_lst']
driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in driving_rgb_crop_lst]

log("Making template ...")
c_d_eyes_lst, c_d_lip_lst = pl.live_portrait_wrapper.calc_ratio(driving_lmk_crop_lst)
# save the motion template
driving_template_dct = make_template(pl, driving_rgb_crop_256x256_lst, c_d_eyes_lst, c_d_lip_lst, output_fps=output_fps)

wfp_template = remove_suffix(args.driving) + '.pkl'
dump(wfp_template, driving_template_dct)
