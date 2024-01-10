"""Callbacks utils."""
from typing import List

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Callback


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiate callbacks from config.

    Args:
    ----
        callbacks_cfg (DictConfig): Config with callbacks.

    Returns:
    -------
        List[Callback]: List of callbacks.
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks
