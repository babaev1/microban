# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright 2026 Marc Duclusaud

"""Read a policy's training-time per-channel observation mean/std straight out of its
own exported .onnx file, for moves/walk.py's out-of-domain (OOD) detector.

`EmpiricalNormalization` (the obs normalizer rsl_rl trains alongside the policy)
accumulates a running per-channel mean/std over every observation seen during
training, and exports as `obs_normalizer._mean` plus one more initializer feeding a
`Sub(obs, mean) -> Div(..., std)` pair right before the first Linear layer — see any
`walk*.onnx`'s graph. onnxruntime.InferenceSession has no API for initializer *values*
(only shapes and the string custom_metadata_map default_joint_pos/action_scale/etc.
already read elsewhere), so getting the actual numbers means loading the raw onnx
protobuf graph instead, which is what this module is for.

The std tensor's name is NOT hardcoded (seen as "onnx::Div_24" in current exports) —
that's an auto-generated name from the PyTorch->ONNX exporter and can change on a
future re-export (a different op count, a different exporter version, ...). Instead
this walks the graph structurally: find the Sub node that consumes
"obs_normalizer._mean", then the Div node that consumes that Sub's output, then take
whichever of ITS inputs isn't the Sub output — the std tensor, by construction,
regardless of what it happens to be named this time.
"""

from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def load_obs_normalizer_stats(onnx_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std), each shape (obs_dim,), read from `onnx_path`'s own graph.

    Raises RuntimeError if the expected obs_normalizer Sub->Div pattern isn't found —
    this should only happen if a checkpoint is exported without EmpiricalNormalization
    at all, in which case the caller has no stats to build an OOD detector from and
    needs to know that explicitly rather than silently get garbage.
    """
    model = onnx.load(str(onnx_path))
    graph = model.graph
    initializers = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

    mean = initializers.get("obs_normalizer._mean")
    if mean is None:
        raise RuntimeError(
            f"{onnx_path}: no 'obs_normalizer._mean' initializer found — this export "
            "has no EmpiricalNormalization stats to build an OOD detector from."
        )

    sub_output = None
    for node in graph.node:
        if node.op_type == "Sub" and "obs_normalizer._mean" in node.input:
            sub_output = node.output[0]
            break
    if sub_output is None:
        raise RuntimeError(
            f"{onnx_path}: found 'obs_normalizer._mean' but no Sub node consuming it — "
            "unexpected graph shape, can't locate the matching std tensor."
        )

    std = None
    for node in graph.node:
        if node.op_type == "Div" and sub_output in node.input:
            std_name = next(n for n in node.input if n != sub_output)
            std = initializers.get(std_name)
            break
    if std is None:
        raise RuntimeError(
            f"{onnx_path}: found the obs_normalizer Sub node but no following Div "
            "node (or its other input isn't an initializer) — unexpected graph shape."
        )

    return mean.flatten().astype(np.float64), std.flatten().astype(np.float64)
