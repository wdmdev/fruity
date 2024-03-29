"""Training script for the model."""
from typing import List, Optional, Tuple, Mapping, Any

import hydra
import wandb
import torch
import omegaconf
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer

from fruity import utils


def train(cfg: DictConfig) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during training.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
    ----
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
    -------
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    cfg.model.net.num_classes = cfg.datamodule.num_classes
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    model: LightningModule = hydra.utils.instantiate(cfg.model)

    logging: LightningModule = hydra.utils.instantiate(cfg.logging)

    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "trainer": trainer,
        "logging": logging,
        # "metrics": train_metrics,
    }

    if cfg.get("train"):
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    # save the weights of the best model after training
    check_point = torch.load(trainer.checkpoint_callback.best_model_path, map_location="cpu")
    state_dict = check_point["state_dict"]
    state_dict = {k.replace("net.", ""): v for k, v in state_dict.items()}
    torch.save(state_dict, cfg.model.net.model_name + "_" + cfg.datamodule.dataset_name + ".pth")

    if cfg.get("test"):
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.2", config_path="../../conf", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Constitutes the main entry point for training.

    Args:
    ----
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
    -------
        Optional[float]: Optimized metric value.
    """
    # Do wandb logging if the run flag is set

    if cfg.logging.run:
        # with open(hydra.utils.to_absolute_path("conf/train.yaml"), "r") as file:
        #     wandb_config = yaml.safe_load(file)
        _ = wandb.init(entity=cfg.logging.entity, project=cfg.logging.project)
        wandb.config.update(omegaconf.OmegaConf.to_container(cfg, resolve=True))

    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    if cfg.logging.run:
        wandb.log(metric_dict)

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
