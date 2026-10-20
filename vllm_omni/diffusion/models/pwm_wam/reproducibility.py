# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Explicit, opt-in controls for a dedicated numerical-validation worker."""

import os
import random

import numpy as np
import torch


def configure_validation(seed: int) -> dict:
    """Seed available RNGs and fail on known nondeterministic Torch operations.

    External FA3 operators are opaque to Torch's check. Repeated full requests
    are still required; these settings do not promise cross-version equality.
    Set the cuBLAS workspace environment before starting the worker.
    """
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in (":4096:8", ":16:8"):
        raise ValueError("Deterministic validation requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "seed": seed,
        "torch": torch.__version__,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": workspace,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "external_kernel_determinism": "must_check_full_request_repeats",
        "latent_rng": "independently_seeded_numpy_RandomState_per_modality",
    }
