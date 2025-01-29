import torch

from .config import Config

class MLP(torch.nn.Module):
    def __init__(
        self,
        dim_in:int,
        dim_out:int,
        depth:int,
        width:int,
        act_fn=torch.nn.ReLU,
        act_fn_out=None,
    ):
        super().__init__()

        layers = []

        self.dim_in = dim_in
        self.dim_out = dim_out

        for layernum in range(depth):

            layers.append(torch.nn.Linear(
                in_features=dim_in if layernum == 0 else width,
                out_features=dim_out if layernum == depth - 1 else width,
            ))

            if layernum < depth - 1:
                layers.append(act_fn())
            elif act_fn_out is not None:
                layers.append(act_fn_out())

        self.layers = torch.nn.Sequential(*layers)

    @property
    def device(self):
        return next(iter(self.layers[0])).device

    def forward(self, x):
        return self.layers(x)

class AudioToMotionModel(MLP):
    def __init__(
        self,
        cfg:Config,
    ):
        self.cfg = cfg
        super().__init__(
            dim_in=cfg.dim_aud,
            dim_out=cfg.num_verts*3,
            depth=cfg.model_depth,
            width=cfg.model_width,
            act_fn_out=None,
        )

    def forward(self, aud_feat:torch.Tensor) -> torch.Tensor:
        bs = aud_feat.shape[0]
        pred = super().forward(aud_feat)

        return pred.reshape(bs, -1, 3)
