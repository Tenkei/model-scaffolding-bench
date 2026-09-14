"""Benchmark PaddleOCR text detection's native path loading against concurrent decoding.

Install the optional runtime dependencies first:

    python -m pip install paddleocr paddlepaddle

Then run from the repository root:

    # Smoke test: three path/decoder routes for one document.
    python -m ml_pipes benchmark src.run_paddleocr_batch_decode_comparison \
        --axis strategy=paddleocr-paths,scatter-paddleocr-decode,scatter-decode \
        --axis max_concurrency=1 \
        --data-axis batch_size=1 \
        --runs 1 --warmup 0

The tiny text detector is intentionally used instead of PaddleOCR's full
five-model OCR pipeline, whose compute time hides input-stage effects. The
benchmark CLI otherwise defaults to 100 measurements plus 10 warmups per
cell, so always set ``--runs`` and ``--warmup`` explicitly.

``batch_size`` repeats one source path so every cell processes the same image
the requested number of times. Override it with
``--data-axis source=/data/document.png``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal
from urllib.request import urlretrieve

import numpy as np
import numpy.typing as npt
from PIL import Image

try:
    from paddleocr import TextDetection
    from paddlex.inference.common.reader import ReadImage
    from paddlex.inference.models.text_detection.result import TextDetResult
except ModuleNotFoundError as exc:  # pragma: no cover - depends on an optional runtime.
    raise SystemExit(
        "This example requires PaddleOCR. Install it with:\n"
        "  python -m pip install paddleocr paddlepaddle"
    ) from exc

from ml_pipes.core import Pipeline
from ml_pipes.factory import InputFn, data_factory, pipeline_factory
from ml_pipes.operator import Operator
from ml_pipes.standard import Gather, Map, Scatter
from ml_pipes.vision import Decode, ImagePayload, LoadFile

EXAMPLE_ASSETS = Path(__file__).parent / ".example_assets"
PADDLEOCR_SAMPLE_URL = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png"
)
LARGE_SAMPLE_SIZE = (4096, 2414)


def default_image() -> Path:
    """Return a 4K JPEG derived once from PaddleOCR's documented OCR sample.

    The upstream sample is only 896 by 528 pixels, too small for a decoding
    comparison to be meaningful. The enlarged cached JPEG makes image I/O
    measurable; pass ``--data-axis source=...`` to benchmark a real large
    document from the deployment instead.
    """
    source = EXAMPLE_ASSETS / "paddleocr-sample.jpg"
    if not source.exists():
        EXAMPLE_ASSETS.mkdir(parents=True, exist_ok=True)
        urlretrieve(PADDLEOCR_SAMPLE_URL, source)

    image = EXAMPLE_ASSETS / "paddleocr-sample-4k.jpg"
    if not image.exists():
        with Image.open(source) as original:
            original.resize(LARGE_SAMPLE_SIZE, Image.Resampling.LANCZOS).save(
                image, quality=95
            )
    return image


def image_array(payload: ImagePayload) -> npt.NDArray[np.uint8]:
    """Extract the BGR array expected by PaddleOCR's NumPy input boundary."""
    return payload.array


@Operator
class TextDetectionPredict:
    """A pipeline boundary around :meth:`paddleocr.TextDetection.predict`."""

    def __init__(self, model: TextDetection, *, batch_size: int = 1) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.model = model
        self.batch_size = batch_size

    def __call__(self, source: list[str] | list[npt.NDArray[np.uint8]]) -> list[TextDetResult]:
        return self.model.predict(input=source, batch_size=self.batch_size)


@Operator
class PaddleOCRDecode:
    """Decode one path to PaddleOCR's BGR NumPy input format."""

    def __init__(self) -> None:
        self.reader = ReadImage(format="BGR")

    def __call__(self, source: str) -> npt.NDArray[np.uint8]:
        return self.reader.read(source)


@pipeline_factory
def build_pipeline(
    strategy: Literal[
        "paddleocr-paths", "scatter-paddleocr-decode", "scatter-decode"
    ] = "paddleocr-paths",
    model_name: str = "PP-OCRv6_tiny_det",
    device: str | None = None,
    max_concurrency: int = 4,
) -> Pipeline[list[str], list[TextDetResult]]:
    """Build one benchmark variant; model setup is excluded from measured calls."""
    model = TextDetection(model_name=model_name, device=device)
    predict = TextDetectionPredict(model)
    if strategy == "paddleocr-paths":
        return Pipeline([predict], auto_validate=True)
    if strategy == "scatter-paddleocr-decode":
        return Pipeline(
            [
                Scatter(max_concurrency=max_concurrency),
                PaddleOCRDecode(),
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
    source: str | None = None,
) -> InputFn:
    """Build a stable batch by repeating one document image path."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    source = str(default_image()) if source is None else source
    if not Path(source).is_file():
        raise ValueError(f"source is not a file: {source!r}")
    batch = [source] * batch_size

    def input_fn() -> tuple[str, list[str], None, dict[str, int]]:
        return (f"batch-{batch_size}", batch, None, {"batch_size": batch_size})

    return input_fn
