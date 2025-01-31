from pathlib import Path
import imageio
import tyro
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig

import cv2
import numpy as np
from rich.progress import track

import torch

from src.live_portrait_wrapper import LivePortraitWrapper
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.utils.cropper import Cropper
from src.utils.camera import get_rotation_matrix
from src.utils.io import dump



def load_video(video_info, n_frames=-1):
    reader = imageio.get_reader(video_info, 'ffmpeg')
    fps = reader.get_meta_data().get('fps', 25.)

    frames = []
    for idx, frame_rgb in enumerate(reader):
        if n_frames > 0 and idx >= n_frames:
            break
        frames.append(frame_rgb)

    reader.close()
    return fps,frames


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

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

def make_template(live_portrait_wrapper:LivePortraitWrapper, driving_rgb_crop_256x256_lst, c_eyes_lst, c_lip_lst, **kwargs):
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
        I_i = prepare_frame(driving_rgb_crop_256x256_lst[i], device=live_portrait_wrapper.device)
        x_i_info = live_portrait_wrapper.get_kp_info(I_i)
        x_s = live_portrait_wrapper.transform_keypoint(x_i_info)
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

def main(
    driver:Path,
):

    print("Setting up config ...")
    args = ArgumentConfig(
        source=None,driving=None, # Prevent stupid mistakes with default arguments
    )

    assert driver.is_file(), f"Cannot find {driver}"
    assert driver.suffix == '.mp4', f"Unsupported filetype {driver.suffix}"

    save_to = driver.with_name('motion.pkl')
    # if save_to.is_file():
    #     print(f"Output file {save_to} already exists, stopping ...")
    #     return

    # specify configs for inference
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    print("Setting up pipeline ...")
    cropper = Cropper(crop_cfg=crop_cfg)
    live_portrait_wrapper = LivePortraitWrapper(inference_cfg=inference_cfg)

    print("Reading input ...")
    output_fps, rgb_lst = load_video(driver)
    num_frames = len(rgb_lst)
    vid_dur = num_frames / output_fps
    print(f"Input has {num_frames} frames at {output_fps} FPS, so input video has length {vid_dur//60: 2.0f}m {vid_dur%60: 2.0f}s {(vid_dur%1)*1000: 3.0f}ms")

    print("Cropping video ...")
    cropper_results = cropper.crop_driving_video(rgb_lst)

    rgb_lst_256p = [cv2.resize(_, (256, 256)) for _ in cropper_results['frame_crop_lst']]

    print("Making template ...")
    c_eyes_lst, c_lip_lst = live_portrait_wrapper.calc_ratio(cropper_results['lmk_crop_lst'])
    # save the motion template
    driving_template_dct = make_template(live_portrait_wrapper, rgb_lst_256p, c_eyes_lst, c_lip_lst, output_fps=output_fps)

    dump(str(save_to), driving_template_dct)

if __name__ == '__main__':
    import tyro
    from src.utils.rprint import rlog as print
    tyro.extras.set_accent_color("bright_cyan")

    tyro.cli(main)
