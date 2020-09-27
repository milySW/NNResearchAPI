import logging

import numpy as np
import torch


def set_deterministic_environment(manual_seed: int, logger: logging.Logger):
    logger.info(f"Seed the RNG for all devices with {manual_seed}")
    torch.manual_seed(manual_seed)
    # see https://pytorch.org/docs/stable/notes/randomness.html
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(manual_seed)
