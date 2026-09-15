# Methodology

This study asks how much end-to-end inference performance depends on the path
that carries an input to a fixed model. Each benchmark compares the provider's
documented, convenient file-input façade with an equivalent pipeline that makes
input loading, preprocessing, batching, and scheduling explicit.

The comparison never changes the checkpoint, task, input pixels, color
semantics, model-facing batch size, or result contract. It measures the
surrounding inference software, not model quality or accuracy.

## Framework selection: ml-pipes

The study uses [ml-pipes](https://github.com/trained-by-humans/ml-pipes)
because the experiment framework needs the following capabilities:

- Batching and concurrency primitives for making scheduling an explicit
  experimental variable.
- Standard load, decode, mapping, and related input-preparation operators
  needed for model inference pipelines.
- Composability: the same pipeline structure can be reused across experiments
  and model providers.
- A lightweight integration surface, so a model's native API can be wrapped
  with minimal provider-specific code.
- Stage-level tracing, allowing end-to-end and individual-stage performance to
  be analyzed.
- Benchmarking tools that run parameterized experiments and save results
  without custom measurement boilerplate.

## Model selection

Case studies are chosen for their inference interface, not simply their model
quality or benchmark popularity.

### Required

- The provider offers a documented, convenient inference façade that accepts
  source-like inputs, such as file paths.
- The provider also exposes a documented boundary for decoded arrays or
  model-ready tensors.
- Both entry points can preserve the same checkpoint, task, input semantics,
  requested outputs, and model-facing batch size.
- The model supports batch inference, so preparation scheduling can be measured
  against a shared model-execution stage.

### Desired

- The model is widely used or representative of a major ecosystem and covers a
  common inference workload.
- The preparation-to-inference time ratio is meaningful enough to measure the
  effect of input-pipeline changes.
- The model is practical to run in the available CPU and GPU environments.

## Experimental ladder

Each case study begins with the native façade, then progressively moves its
input stages into an explicit pipeline. The diagram is read from top to bottom;
the numbered arrows show the order in which stages are extracted.

```text
Native model scaffolding                         Optimized pipeline
────────────────────────                         ──────────────────
source paths                                     source paths
    │                                                │
    ▼                                                ▼
provider loads source                 [1] ───►  Load source
    │                                                │
    ▼                                                ▼
provider decodes input                [2] ───►  Decode
    │                                                │
    ▼                                                ▼
provider preprocesses input           [3] ───►  Preprocess
    │                                                │
    ▼                                                ▼
provider invokes the model                        Model inference
    │                                                │
    ▼                                                ▼
provider postprocesses output                     Postprocess
    │                                                │
    ▼                                                ▼
result contract                                  same result contract
```

An intermediate rung extracts only the first one or more numbered stages; the
provider continues to perform the remaining stages. The final route exposes all
available preparation stages and invokes the same underlying model directly.
This ladder attributes a performance change to a specific execution boundary
rather than treating an explicit pipeline as one opaque alternative.

## Measurement protocol

- Keep model construction, download, and first-use compilation outside the
  measured section.
- Treat cold-cache and steady-state behavior as separate experiments.
- Warm up each configuration, then report mean, p50, p95, and p99 latency as
  well as images per second.
- Randomize or interleave configuration order to reduce CPU frequency,
  thermal-state, and background-load bias.
- Capture CPU utilization, GPU utilization, memory, and stage-level timings
  alongside total latency.
- Run a correctness check on every route before interpreting a timing result.

## Next

Read [Running experiments](running-experiments.md) to execute a case study and
save its measurements.
