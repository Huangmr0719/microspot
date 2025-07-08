# -*- coding: utf-8 -*-
"""
python train.py trainer.max_epochs=50 model.lr=3e-4
"""
import hydra, pytorch_lightning as pl, torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback

# local imports
from src.data.datamodule import MicroSpotDataModule
from src.models.lit_module import MicroSpotLit

@hydra.main(version_base=None, config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig):

    # reproducibility
    pl.seed_everything(cfg.seed, workers=True)

    # -------- build objects w/ Hydra ---------------------------------
    dm     = MicroSpotDataModule(cfg.datamodule)
    model  = hydra.utils.instantiate(cfg.model)

    #   instantiate loggers & callbacks list
    loggers   = [hydra.utils.instantiate(l) for l in cfg.logger]
    callbacks = [hydra.utils.instantiate(c) for c in cfg.callbacks]

    # -------- trainer ------------------------------------------------
    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger    = loggers,
        callbacks = callbacks
    )

    # -------- training / test ---------------------------------------
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm, ckpt_path="best")

if __name__ == "__main__":
    main()