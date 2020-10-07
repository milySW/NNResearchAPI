from __future__ import annotations

import logging

import torch

from pytorch_lightning import seed_everything

import configs


def set_deterministic_environment(manual_seed: int, logger: logging.Logger):
    logger.info(f"Seed the RNG for all devices with {manual_seed}")
    # see https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_everything(manual_seed)


def setup(train_config: configs.DefaultTraining, logger: logging.Logger):
    if train_config.test and not train_config.save:
        logger.warn("Test pipeline is not possible while save param = False")

    if train_config.seed is not None:
        set_deterministic_environment(train_config.seed, logger)
