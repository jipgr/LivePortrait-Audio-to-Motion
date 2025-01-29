from pathlib import Path
from typing import Callable
import numpy as np
import torch
from torch.optim import Optimizer,Adam,lr_scheduler
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

import tyro

from src.a2m.config import Config
from src.a2m.dataset import AudioToMotionDataset
from src.a2m.model import AudioToMotionModel
from src.utils.rprint import rprint as print,rlog as log
from src.live_portrait_pipeline_a2m import LPPipeA2M

step=0

def train(
    model:AudioToMotionModel,
    optim:Optimizer,
    loss_fn:Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    loader:DataLoader,
    device:torch.device,
    writer:SummaryWriter|None=None,
):
    global step
    model.train()

    loss_acum,loss_norm = 0., 0
    for x, y in loader:
        optim.zero_grad()

        x:torch.Tensor = x.to(device)
        y:torch.Tensor = y.to(device)

        yhat = model(x)

        loss = loss_fn(y, yhat)

        loss.backward()

        optim.step()

        if writer is not None:
            step += 1
            writer.add_scalar('train/loss', loss.item(), step)

        loss_acum += loss.item()
        loss_norm += yhat.shape[0]

    return 0. if loss_norm == 0 else loss_acum / loss_norm

def validate(
    model:AudioToMotionModel,
    loss_fn:torch.nn.Module,
    loader:DataLoader,
    device:torch.device,
    writer:SummaryWriter|None=None,
):
    global step
    model.eval()

    loss_acum,loss_norm = 0., 0
    for aud_feat, motion_gt in loader:
        aud_feat:torch.Tensor = aud_feat.to(device)
        motion_gt:torch.Tensor = motion_gt.to(device)

        with torch.no_grad():
            motion_pred = model(aud_feat)

        loss:torch.Tensor = loss_fn(motion_gt, motion_pred)

        loss_acum += loss.item()
        loss_norm += motion_pred.shape[0]

    loss_total = 0. if loss_norm == 0 else loss_acum / loss_norm
    writer.add_scalar('eval/loss', loss_total, step)

    return loss_total


def main(cfg:Config):
    log(f"Load data from {cfg.data_root}")
    traindata = AudioToMotionDataset(cfg, 'train')
    trainloader = DataLoader(
        dataset=traindata,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_worker
    )

    valdata = AudioToMotionDataset(cfg, 'val')
    valloader = DataLoader(
        dataset=valdata,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_worker
    )

    log("Setting up Audio-to-Motion model ...")
    model = AudioToMotionModel(cfg)

    model.to(cfg.device)

    if cfg.load_ckpt is not None:
        model.load_state_dict(torch.load(
            cfg.load_ckpt,
            map_location=cfg.device
        ))

    loss_fn = torch.nn.L1Loss()


    optim = Adam(
        params=model.parameters(),
        lr=cfg.lr,
    )

    schedule = lr_scheduler.MultiplicativeLR(
        optimizer=optim,
        lr_lambda=lambda epoch: np.exp(np.log(.1)/cfg.num_epoch),
    )

    writer = SummaryWriter(cfg.model_dir)

    cfg.model_dir.mkdir(parents=True, exist_ok=True)

    log("Training ...")
    for epoch in range(1,cfg.num_epoch+1):
        epoch_loss = train(
            model=model,
            optim=optim,
            loss_fn=loss_fn,
            loader=trainloader,
            device=cfg.device,
            writer=writer,
        )
        # print(f"Epoch {epoch:04d}/{cfg.num_epoch:04d}: train loss={epoch_loss:.2e}")

        schedule.step()
        if writer is not None:
            writer.add_scalar('train/lr', schedule.get_last_lr()[0], step)

        if epoch == 1 or epoch % cfg.eval_every == 0:
            epoch_loss = validate(
                model=model,
                loss_fn=loss_fn,
                loader=valloader,
                device=cfg.device,
                writer=writer,
            )
            log(f"Epoch {epoch:04d}/{cfg.num_epoch:04d}: val loss={epoch_loss:.2e}")

        if epoch % cfg.ckpt_every == 0:
            save_to = cfg.model_dir / f'model_ep{epoch}.pth'
            torch.save(model.state_dict(), save_to)

    log("Done.")

if __name__ == '__main__':
    cfg = tyro.cli(Config)
    tyro.extras.set_accent_color("bright_cyan")
    main(cfg)
