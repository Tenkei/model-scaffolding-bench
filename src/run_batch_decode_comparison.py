"""Benchmark native path loading against concurrent ml-pipes image decoding.

Run from the repository root with the ml-pipes benchmark CLI:

    python -m ml_pipes benchmark src.run_batch_decode_comparison \
        --axis strategy=ultralytics-paths,scatter-ultralytics-decode,scatter-decode \
        --data-axis batch_size=1,4,8

``batch_size`` repeats one source path so each cell processes the same image
the requested number of times. Override that source with
``--data-axis source=/data/image.jpg``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from ultralytics.data.loaders import LoadPilAndNumpy, autocast_list
from ultralytics.engine.results import Results

from ml_pipes.core import Pipeline
from ml_pipes.factory import InputFn, data_factory, pipeline_factory
from ml_pipes.operator import Operator
from ml_pipes.standard import Gather, Map, Scatter
from ml_pipes.ultralytics import yolo
from ml_pipes.vision import Decode, ImagePayload, LoadFile


def image_array(payload: ImagePayload) -> npt.NDArray[np.uint8]:
    """Extract the BGR array that YOLO accepts from a decoded image payload."""
    return payload.array


@Operator
class UltralyticsDecode:
    """Decode a path through the same two upstream stages as a YOLO list source."""

    def __call__(self, source: Path) -> npt.NDArray[np.uint8]:
        image = autocast_list([source])
        return LoadPilAndNumpy(image).im0[0]


@pipeline_factory
def build_pipeline(
    strategy: Literal[
        "ultralytics-paths", "scatter-ultralytics-decode", "scatter-decode"
    ] = "ultralytics-paths",
    model: str = "yolo26n.pt",
    conf: float = 0.25,
    imgsz: int = 640,
    device: str | None = None,
    max_concurrency: int = 4,
) -> Pipeline[list[Path], list[Results]]:
    """Build one benchmark variant; pipeline construction is outside measured runs."""
    predict = yolo.Predict(model=model, conf=conf, imgsz=imgsz, device=device)
    if strategy == "ultralytics-paths":
        return Pipeline([predict], auto_validate=True)
    if strategy == "scatter-ultralytics-decode":
        return Pipeline(
            [
                Scatter(max_concurrency=max_concurrency),
                UltralyticsDecode(),
                Gather(),
                predict,
            ],
            auto_validate=True,
        )
    return Pipeline(
        [
            Scatter(max_concurrency=max_concurrency),
            LoadFile(),
            Decode(),
            Map(image_array),
            Gather(),
            predict,
        ],
        auto_validate=True,
    )


@data_factory
def build_batch_input(
    batch_size: int = 1,
    source: str = "assets/images/ultralytics-sample.jpg",
) -> InputFn:
    """Build a stable batch by repeating one image path."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    image = Path(source)
    if not image.is_file():
        raise ValueError(f"source is not a file: {source!r}")
    batch = [image] * batch_size

    def input_fn() -> tuple[str, list[Path], None, dict[str, int]]:
        return (f"batch-{batch_size}", batch, None, {"batch_size": batch_size})

    return input_fn
