# model-scaffolding-bench

LLMs made batching an obvious optimization. But in image and document
workloads, a convenient call such as `model(paths)` can still hide source
loading, image decoding, and preprocessing that run serially while a paid GPU
waits for the next batch.

`model-scaffolding-bench` studies those hidden pipeline decisions. It compares
a model provider's documented, path-based inference façade with equivalent
pipelines that make loading, decoding, preprocessing, batching, device
transfer, and scheduling explicit.

Every comparison keeps the checkpoint, input pixels, task, and requested
outputs fixed. The goal is to measure how much end-to-end efficiency can be
recovered by exposing the pipeline around a fixed model—not to claim that the
model itself became faster.

## Project structure

```text
assets/         Versioned source inputs shared by case studies.
src/            Runnable framework-specific case studies and reusable code.
docs/           Methodology and provider setup guides.
environments/   Immutable environment definitions used for reported runs.
results/        Experiment output, environment captures, and derived data.
```

## Methodology

Each case study begins with the inference façade exactly as a typical user would
use it: follow the provider's documented setup and pass source inputs to its
convenient API, without assuming detailed knowledge of its concurrency,
timing, batching, or resource behavior.

It then progressively exposes and instruments the source loading,
preprocessing, batching, and model-invocation boundaries hidden by that façade.
This shows how much efficiency can be recovered by making pipeline execution
explicit, while preserving the model, task, input semantics, and requested
output.

See [Methodology](docs/methodology.md) for the complete measurement protocol.

## Current case studies

| Framework                 | Model / Task                 | Upstream reference                                                                               | Benchmark                                             |
|---------------------------|------------------------------|--------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| Ultralytics               | YOLO detection               | [Predict mode](https://docs.ultralytics.com/modes/predict/)                                      | `src/run_batch_decode_comparison.py`                  |
| PaddleOCR                 | PP-OCRv6 tiny text detection | [Text Detection](https://www.paddleocr.ai/latest/en/version3.x/module_usage/text_detection.html) | `src/run_paddleocr_batch_decode_comparison.py`        |
| Hugging Face Transformers | ViT image classification     | [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)                | `src/run_transformers_vit_batch_decode_comparison.py` |

See the [Case-study index](docs/index.md) for installation commands,
pipeline variants, supported axes, default inputs, and example runs.

The examples use [ml-pipes](https://github.com/trained-by-humans/ml-pipes) to
make decode, preprocessing, and batching stages observable and independently
scheduled. They retain each upstream framework at its documented native input
boundary whenever possible.

## Documentation

- [Study scope](docs/scope.md) — the optimization patterns and batch-processing
  use case covered by the project.
- [Methodology](docs/methodology.md) — experimental design and measurement
  protocol.
- [Case-study index](docs/index.md) — models, dependencies, pipeline variants,
  supported axes, and example commands.
- [Running experiments](docs/running-experiments.md) — provider preparation,
  repeatable comparisons, and result storage.
- [Viewing stored results](docs/viewing-results.md) — result artifacts and
  programmatic loading.
