
import os
import random

import numpy as np
import torch


def set_seed(seed=42, deterministic=False):
    """
    Set random seeds for reproducible experiments.

    Args:
        seed: Random seed.
        deterministic: Request deterministic PyTorch algorithms.
                       May reduce performance or raise errors if an
                       operation has no deterministic implementation.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Must be set before CUDA context initialization for full effect.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)

