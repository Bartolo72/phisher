import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from typing import List

from phisher.model import PhisherModel
from phisher.module import PhisherhModule
from phisher.dataset import PhishingDataModule, PhishingDataset
from phisher.dataset.utils import prepare_phish_dataset


console = Console()

@hydra.main(config_path="conf", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    console.log("[bold gold3]Starting dataset preparation...[/bold gold3]")
    
    with Progress(
        SpinnerColumn(spinner_name="dots", style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Preparing dataset...", total=len(cfg.data))
        for idx, data_config in enumerate(cfg.data.values()):
            prepare_phish_dataset(target_file_path="phish_dataset.csv", **data_config)
            progress.update(task, advance=1)
    console.log("[bold green]Dataset prepared successfully.[/bold green]")
    
    console.log("[bold gold3]Initializing components...[/bold gold3]")

    dataset: PhishingDataset = instantiate(cfg.dataset)
    console.log(f"Dataset instantiated: [green]{type(dataset).__name__}[/green]")
    
    data_module: PhishingDataModule = instantiate(cfg.data_module, dataset=dataset)
    data_module.setup(**cfg.data_module_setup)
    console.log(f"DataModule instantiated: [green]{type(data_module).__name__}[/green]")
    
    model: PhisherModel = instantiate(cfg.model)
    console.log(f"Model instantiated: [green]{type(model).__name__}[/green]")
    
    optimizer: torch.optim.Optimizer = instantiate(cfg.optimizer, params=model.parameters())
    console.log(f"Optimizer instantiated: [green]{type(optimizer).__name__}[/green]")
    
    module = PhisherhModule(model, optimizer, num_classes=2)
    console.log(f"Lightning Module instantiated: [green]{type(module).__name__}[/green]")
    
    callbacks: List[Callback] = [instantiate(cb_cfg) for cb_cfg in cfg.callbacks.values()]
    console.log(f"Callbacks: [green]{[type(cb).__name__ for cb in callbacks]}[/green]")
    
    logger = instantiate(cfg.logger)
    console.log(f"Logger: [green]{type(logger).__name__}[/green]")
    
    trainer: pl.Trainer = pl.Trainer(
        **cfg.trainer,
        logger=logger,
        callbacks=callbacks,
    )
    console.log("[bold gold3]Training started...[/bold gold3]")
    trainer.fit(module, data_module)
    console.log("[bold gold3]Training finished.[/bold gold3]")
    trainer.test(model=module, datamodule=data_module)
    console.log("[bold gold3]Testing finished.[/bold gold3]")


if __name__ == "__main__":
    main()
