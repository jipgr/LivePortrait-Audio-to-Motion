# coding: utf-8

"""
Pipeline of LivePortrait (Human)
"""

import dataclasses
import torch

import cv2

from .config.argument_config import ArgumentConfig
from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .utils.cropper import Cropper
from .utils.camera import get_rotation_matrix
from .utils.io import load_image_rgb, resize_to_limit
from .live_portrait_wrapper import LivePortraitWrapper
from .a2m.config import REGION_TO_VERTICES


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


@dataclasses.dataclass
class SourceMotion:
    R:torch.Tensor # 1,3,3
    t:torch.Tensor # 1,3
    scale:torch.Tensor # 1,1
    kp:torch.Tensor # 1,21,3
    exp:torch.Tensor # 1,21,3
    f_s:torch.Tensor # 1,32,16,64,64
    x_s:torch.Tensor # 1,21,3

class LPPipeA2M(object):
    """LivePortrait PipeLine for Audio-to-Motion"""

    def __init__(self, **kwargs):
        kwargs['driving'] = None # prevent accidentally loading driving parameters
        self.cfg = ArgumentConfig(**kwargs)

        self.live_portrait_wrapper = LivePortraitWrapper(
            inference_cfg=partial_fields(InferenceConfig, self.cfg.__dict__)
        )
        self.cropper: Cropper = Cropper(
            crop_cfg=partial_fields(CropConfig, self.cfg.__dict__)
        )

        self.source = self.prep_source()
        self.motion_multiplier = 1.

    def prep_source(self, fp=None):
        if fp is None:
            fp = self.cfg.source

        img_rgb = load_image_rgb(fp)
        img_rgb = resize_to_limit(img_rgb, self.cfg.source_max_dim, self.cfg.source_division)

        crop_info = self.cropper.crop_source_image(img_rgb, self.cropper.crop_cfg)
        if crop_info is None:
            raise Exception("No face detected in the source image!")

        I_s = self.live_portrait_wrapper.prepare_source(crop_info['img_crop_256x256'])
        x_s_info = self.live_portrait_wrapper.get_kp_info(I_s)

        return SourceMotion(
            R=get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll']),
            t=x_s_info['t'],
            scale=x_s_info['scale'],
            kp=x_s_info['kp'],
            exp=x_s_info['exp'],
            f_s=self.live_portrait_wrapper.extract_feature_3d(I_s),
            x_s=self.live_portrait_wrapper.transform_keypoint(x_s_info),
        )

    def render(self, exp_delta:torch.Tensor, region:str|None=None):
        """
        Render for a given expression delta

        Args:
        ----
        - :param Tensor exp_delta: Shape V,3 or B,V,3 with either V=21 or V
            matches the number of vertices for the given animation region

        Returns:
        ----
        :returns Tensor: Image, B,3,H,W
        """

        if region is None:
            region = self.cfg.animation_region

        # Add batch dim
        if exp_delta.ndim == 2:
            exp_delta = exp_delta.unsqueeze(0)

        bs = exp_delta.shape[0]

        # Zero out vertices we dont want to focus on
        exp = torch.zeros(bs, *self.source.exp.shape[1:], dtype=self.source.exp.dtype, device=self.source.exp.device)
        idxs = REGION_TO_VERTICES.get(region, [])

        # Exp_delta is a full exp vector -> only take relevant indices
        if exp_delta.shape[1] == exp.shape[1]:
            exp[:, idxs] = exp_delta[:, idxs]

        # Assume exp_delta is already masked out for the relevant indices
        else:
            assert len(idxs) == exp_delta.shape[1], (exp_delta.shape, self.cfg.animation_region)
            exp[:, idxs] = exp_delta

        # Warp keypoints
        x_d_i = self.source.x_s + self.source.scale * self.motion_multiplier * exp

        # Expand source info to match batchsize
        x_s = self.source.x_s.expand(bs, *self.source.x_s.shape[1:])
        f_s = self.source.f_s.expand(bs, *self.source.f_s.shape[1:])

        # Stitch keypoints
        x_d_i = self.live_portrait_wrapper.stitching(x_s, x_d_i)

        # Generate output image
        out = self.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i)

        return out['out']
