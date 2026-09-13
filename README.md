# model-scaffolding-bench

`model-scaffolding-bench` investigates the software surrounding modern model
inference: path loading, image decoding, preprocessing, batch formation,
device transfer, and execution scheduling.

The neural model is often only one part of end-to-end latency. Convenient
framework façades can hide input preparation behind a single call, but that
work may be serialized, repeated per item, or unable to overlap with batched
accelerator inference. This repository measures those effects without changing
the checkpoint, input pixels, task, or requested outputs.

## What it compares

Each benchmark progressively exposes the same model's inference path:

```text
native path inputs
  → concurrent decode before the native boundary
  → explicit batch preprocessing and direct model invocation
  → concurrent decode and preprocessing before batched inference
```

The goal is not to label frameworks as slow. It is to identify when a model
wrapper's execution scaffolding becomes the bottleneck, and to show which
explicit pipeline boundaries recover throughput while preserving results.

## Current case studies

| Framework | Model or task | Benchmark |
|---|---|---|
| Ultralytics | YOLO detection | `examples/run_batch_decode_comparison.py` |
| PaddleOCR | PP-OCRv6 tiny text detection | `examples/run_paddleocr_batch_decode_comparison.py` |
| Hugging Face Transformers | ViT image classification | `examples/run_transformers_vit_batch_decode_comparison.py` |

The examples use [ml-pipes](https://github.com/trained-by-humans/ml-pipes) to
make decode, preprocessing, and batching stages observable and independently
scheduled. They retain each upstream framework at its documented native input
boundary whenever possible.

## Run a benchmark

Install the dependencies required by the selected case study. The project is
designed to be run from a checked-out source tree alongside the framework
integration packages.

For the Hugging Face ViT study:

```bash
python -m pip install transformers

python -m ml_pipes benchmark examples.run_transformers_vit_batch_decode_comparison \
  --axis strategy=transformers-paths,scatter-transformers-decode,direct-model,direct-model-concurrent-preprocess \
  --axis inference_batch_size=8 \
  --axis max_concurrency=8 \
  --data-axis batch_size=8 \
  --runs 20 --warmup 3
```

See [Benchmarking](docs/benchmarking.md) for the experiment protocol and a
CLI-only AWS EC2 workflow.

## Methodology principles

- Keep model checkpoint, input data, image color semantics, batch size, and
  output contract equivalent across routes.
- Separate model construction, downloads, cold-cache behavior, and warm-up
  from steady-state measurements.
- Report end-to-end latency, throughput, per-stage timings, and hardware
  utilization rather than one aggregate number alone.
- Randomize or interleave configuration order to reduce CPU frequency and
  thermal-state bias.
- Validate output equivalence before interpreting a performance difference.

## Scope

This is a research artifact and benchmark suite, not an inference framework.
Framework-specific production operators remain in their own repositories, such
as [ml-pipes-ultralytics](https://github.com/requiem4machines/ml-pipes-ultralytics).

Future case studies and provider guides can be added without coupling them to a
particular framework package or cloud vendor.
