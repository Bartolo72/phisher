import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from typing import List

from phisher.model import PhisherModel
from phisher.module import PhisherhModule
from phisher.dataset import PhishingDataModule, PhishingDataset


@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    dataset: PhishingDataset = instantiate(cfg.dataset) 
    data_module: PhishingDataModule = instantiate(cfg.data_module, dataset=dataset)
    data_module.setup(**cfg.data_module_setup)

    model: PhisherModel = instantiate(cfg.model) 
    optimizer: torch.optim.Optimizer = instantiate(cfg.optimizer, params=model.parameters())
    module = PhisherhModule(model, optimizer, num_classes=2)
    callbacks: List[Callback] = [instantiate(cb_cfg) for cb_cfg in cfg.callbacks.values()]

    logger = instantiate(cfg.logger)

    trainer: pl.Trainer = pl.Trainer(**cfg.trainer, 
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(module, data_module)
    trainer.test(model=module, datamodule=data_module)


if __name__ == "__main__":
    main()

