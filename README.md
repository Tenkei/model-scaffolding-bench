# model-scaffolding-bench

`model-scaffolding-bench` investigates the software surrounding modern model
inference: path loading, image decoding, preprocessing, batch formation,
device transfer, and execution scheduling.

The neural model is often only one part of end-to-end latency. Convenient
framework façades can hide input preparation behind a single call, but that
work may be serialized, repeated per item, or unable to overlap with batched
accelerator inference. This repository measures those effects without changing
the checkpoint, input pixels, task, or requested outputs.

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

## Out of scope

The case studies hold the model export constant. They do not compare
quantization, precision changes, architecture changes, compilation, or
hardware-specific exports such as TensorRT, ONNX Runtime, Core ML, or OpenVINO.

Those techniques can improve model execution and should be evaluated separately.
The pipeline efficiencies measured here are orthogonal: the same source loading,
preprocessing, batching, and scheduling improvements can be applied around any
fixed compatible model export.

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

- [Methodology](docs/methodology.md) — experimental design and measurement
  protocol.
- [Case-study index](docs/index.md) — models, dependencies, pipeline variants,
  supported axes, and example commands.
- [Running experiments](docs/running-experiments.md) — provider preparation,
  repeatable comparisons, and result storage.
- [Viewing stored results](docs/viewing-results.md) — result artifacts and
  programmatic loading.
