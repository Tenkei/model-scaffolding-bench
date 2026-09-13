"""Benchmark Hugging Face ViT path loading against explicit concurrent preparation.

Install the optional runtime dependency first:

    python -m pip install transformers

Then run from the repository root:

    python -m ml_pipes benchmark examples.run_transformers_vit_batch_decode_comparison \
        --axis strategy=transformers-paths,scatter-transformers-decode,direct-model,direct-model-concurrent-preprocess \
        --axis inference_batch_size=1,8 \
        --axis max_concurrency=1,8 \
        --data-axis batch_size=1,8 \
        --runs 20 --warmup 3

``batch_size`` controls how many copies of the source enter one pipeline call.
``inference_batch_size`` is the Hugging Face model batch size, independently.
The default source is a cached 4K image so that input preparation is measurable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from PIL import Image

try:
    from transformers import AutoImageProcessor, ViTForImageClassification, pipeline
    from transformers.image_utils import load_image
except ModuleNotFoundError as exc:  # pragma: no cover - optional runtime dependency.
    raise SystemExit(
        "This example requires Hugging Face Transformers. Install it with:\n"
        "  python -m pip install transformers"
    ) from exc

from ml_pipes.core import Pipeline
from ml_pipes.factory import InputFn, data_factory, pipeline_factory
from ml_pipes.operator import Operator
from ml_pipes.standard import Gather, Scatter

EXAMPLE_ASSETS = Path(__file__).parent / ".example_assets"
SOURCE_IMAGE = Path("docs/assets/bus.jpg")
LARGE_SAMPLE_SIZE = (4096, 5461)

Classification = dict[str, str | float]
Classifications = list[list[Classification]]


def default_image() -> Path:
    """Return a cached 4K image derived from the repository's public bus asset."""
    if not SOURCE_IMAGE.is_file():
        raise FileNotFoundError(f"Example source image not found: {SOURCE_IMAGE}")
    image = EXAMPLE_ASSETS / "vit-sample-4k.jpg"
    if not image.exists():
        EXAMPLE_ASSETS.mkdir(parents=True, exist_ok=True)
        with Image.open(SOURCE_IMAGE) as original:
            original.resize(LARGE_SAMPLE_SIZE, Image.Resampling.LANCZOS).save(image, quality=95)
    return image


@Operator
class TransformersDecode:
    """Load one path through Transformers' own Pillow-based image loader."""

    def __call__(self, source: str) -> Image.Image:
        return load_image(source)


@Operator
class TransformersPipelinePredict:
    """Wrap the native image-classification pipeline and preserve its input boundary."""

    def __init__(
        self,
        classifier: Any,
        *,
        inference_batch_size: int,
        num_workers: int,
    ) -> None:
        self.classifier = classifier
        self.inference_batch_size = inference_batch_size
        self.num_workers = num_workers

    def __call__(self, source: list[str] | list[Image.Image]) -> Classifications:
        return self.classifier(
            source,
            batch_size=self.inference_batch_size,
            num_workers=self.num_workers,
        )


@Operator
class ViTBatchPreprocess:
    """Turn a loaded image batch into ViT's model-ready ``pixel_values`` tensor."""

    def __init__(self, image_processor: Any, *, inference_batch_size: int) -> None:
        self.image_processor = image_processor
        self.inference_batch_size = inference_batch_size

    def __call__(self, images: list[Image.Image]) -> list[torch.Tensor]:
        return [
            self.image_processor(
                images=images[start: start + self.inference_batch_size], return_tensors="pt"
            ).pixel_values
            for start in range(0, len(images), self.inference_batch_size)
        ]


@Operator
class ViTPreprocess:
    """Turn one loaded image into one model-ready ``[C, H, W]`` tensor."""

    def __init__(self, image_processor: Any) -> None:
        self.image_processor = image_processor

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.image_processor(images=image, return_tensors="pt").pixel_values[0]


@Operator
class ViTBatch:
    """Stack individually preprocessed tensors into model inference batches."""

    def __init__(self, *, inference_batch_size: int) -> None:
        self.inference_batch_size = inference_batch_size

    def __call__(self, pixel_values: list[torch.Tensor]) -> list[torch.Tensor]:
        return [
            torch.stack(pixel_values[start: start + self.inference_batch_size])
            for start in range(0, len(pixel_values), self.inference_batch_size)
        ]


@Operator
class ViTPredict:
    """Run the native ViT model directly on a preprocessed batch tensor."""

    def __init__(self, model: ViTForImageClassification, *, device: torch.device) -> None:
        self.model = model
        self.device = device

    def __call__(self, pixel_values: list[torch.Tensor]) -> list[torch.Tensor]:
        # Match ``transformers.Pipeline.forward``: execute without gradients,
        # then move model outputs back to CPU before postprocessing.
        with torch.no_grad():
            return [
                self.model(pixel_values=batch.to(self.device)).logits.cpu()
                for batch in pixel_values
            ]


@Operator
class ViTPostprocess:
    """Match the classification pipeline's top-k label and softmax-score output."""

    def __init__(self, id2label: dict[int, str], *, top_k: int = 5) -> None:
        self.id2label = id2label
        self.top_k = top_k

    def __call__(self, logits: list[torch.Tensor]) -> Classifications:
        classifications: Classifications = []
        for batch in logits:
            scores, indices = torch.softmax(batch, dim=-1).topk(self.top_k, dim=-1)
            for image_scores, image_indices in zip(scores.cpu(), indices.cpu()):
                classifications.append(
                    [
                        {"label": self.id2label[index.item()], "score": score.item()}
                        for score, index in zip(image_scores, image_indices)
                    ]
                )
        return classifications


def resolve_device(device: str | None) -> torch.device:
    """Use CPU by default; accept normal Torch devices such as ``cuda`` or ``mps``."""
    return torch.device("cpu" if device is None else device)


@pipeline_factory
def build_pipeline(
        strategy: Literal[
            "transformers-paths",
            "scatter-transformers-decode",
            "direct-model",
            "direct-model-concurrent-preprocess",
        ] = "transformers-paths",
        model_name: str = "google/vit-base-patch16-224",
        device: str | None = None,
        inference_batch_size: int = 1,
        max_concurrency: int = 4,
) -> Pipeline[list[str], Classifications]:
    """Build one comparison route; model construction is excluded from measurements."""
    if inference_batch_size < 1:
        raise ValueError("inference_batch_size must be at least 1")
    resolved_device = resolve_device(device)

    if strategy == "transformers-paths":
        return Pipeline([
            TransformersPipelinePredict(
                pipeline(
                    "image-classification", model=model_name, device=resolved_device
                ),
                inference_batch_size=inference_batch_size,
                # Transformers creates a nested ``pad_collate_fn`` closure for
                # batched image inputs. It cannot be pickled by macOS's spawn
                # workers, so its native DataLoader must remain single-process.
                # The explicit routes below provide portable parallelism.
                num_workers=0,
            )
        ], auto_validate=True)

    if strategy == "scatter-transformers-decode":
        return Pipeline([
            Scatter(max_concurrency=max_concurrency),
            TransformersDecode(),
            Gather(),
            TransformersPipelinePredict(
                pipeline(
                    "image-classification", model=model_name, device=resolved_device
                ),
                inference_batch_size=inference_batch_size,
                num_workers=0,
            )
        ], auto_validate=True)

    if strategy not in {"direct-model", "direct-model-concurrent-preprocess"}:
        raise ValueError(f"Unsupported strategy: {strategy!r}")

    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name).to(resolved_device).eval()
    if strategy == "direct-model-concurrent-preprocess":
        return Pipeline(
            [
                Scatter(max_concurrency=max_concurrency),
                TransformersDecode(),
                ViTPreprocess(image_processor),
                Gather(),
                ViTBatch(inference_batch_size=inference_batch_size),
                ViTPredict(model, device=resolved_device),
                ViTPostprocess(model.config.id2label),
            ],
            auto_validate=True,
        )

    return Pipeline(
        [
            Scatter(max_concurrency=max_concurrency),
            TransformersDecode(),
            Gather(),
            ViTBatchPreprocess(image_processor, inference_batch_size=inference_batch_size),
            ViTPredict(model, device=resolved_device),
            ViTPostprocess(model.config.id2label),
        ],
        auto_validate=True,
    )


@data_factory
def build_batch_input(batch_size: int = 1, source: str | None = None) -> InputFn:
    """Build a stable path batch by repeating one image source."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    image = default_image() if source is None else Path(source)
    if not image.is_file():
        raise ValueError(f"source is not a file: {str(image)!r}")
    batch = [str(image)] * batch_size

    def input_fn() -> tuple[str, list[str], None, dict[str, int]]:
        return (f"batch-{batch_size}", batch, None, {"batch_size": batch_size})

    return input_fn
