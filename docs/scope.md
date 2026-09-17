# Study scope

This study identifies and measures pipeline patterns that can burn valuable GPU
time without changing the model itself. Its focus is the software around model
execution: loading inputs, decoding them, preprocessing them, forming batches,
and scheduling that work relative to accelerator execution.

## Optimization patterns under study

### Batching as the baseline

Batching is the familiar first optimization lever. Modern LLM workloads have
made its value widely understood: combining compatible requests or inputs lets
one model invocation do more useful work and amortizes fixed overhead.

The same principle applies to image and document workloads. When a user has a
large dataset, feeding one input at a time leaves batching opportunities on the
table. This study keeps the model-facing batch size fixed across routes so it
can distinguish the benefit of batching from the cost of getting inputs ready
for a batch.

### Concurrency as the optimization lever

Concurrency is a standard software-engineering tool, but it is often less
visible in ML workflows. A convenient model call can make source loading,
decode, and preprocessing appear to be part of one indivisible inference
operation. If those CPU-side stages are serialized, the accelerator may wait
for the next model-ready batch.

This study therefore treats preparation concurrency as a first-class variable.
It measures whether making stages explicit allows loading and preprocessing to
overlap or proceed in parallel before batched model execution. It also records
when this does *not* help: the useful optimization boundary depends on the
model provider and workload.

## The representative use-case

This study intentionally focuses on offline batch processing of a large, already-provided dataset—not
the latency or throughput of an online inference API.

In an online service, a backend can often compensate for a provider API's lack
of explicit preparation-concurrency controls by handling multiple requests at
once. Queues, worker pools, and request-level concurrency make that a different
system-design problem. This study instead asks what happens within one batch
job, where a user controls the input list and must prepare work efficiently for
the model.

The representative user journey is deliberately ordinary:

1. A company provides a large dataset and asks a user to run a model over it.
2. The user reads the model documentation, finds the batch API, and learns that
   batches are more efficient than one-item model calls.
3. The user loads a list of input files and sends device-sized groups through
   the model in a loop.

```python
for batch_paths in batched(dataset_paths, batch_size):
    results = model(batch_paths)
```

This is a sensible starting point, but it can hide the work performed before
the model runs. A path-based batch API may still load, decode, or preprocess
items serially, or create batching boundaries that prevent preparation from
overlapping with accelerator work. The case studies compare that baseline with
equivalent routes that expose those boundaries.

## Out of scope

**The case studies hold the model export constant.** They do not compare
quantization, precision changes, architecture changes, compilation, or
hardware-specific exports such as TensorRT, ONNX Runtime, Core ML, or OpenVINO.

Those techniques can improve model execution and should be evaluated separately.
The pipeline efficiencies measured here are orthogonal: the same source loading,
preprocessing, batching, and scheduling improvements can be applied around any
fixed compatible model export.

**Model accuracy is not measured.** Each comparison keeps the checkpoint, task,
input semantics, and requested output contract fixed, so accuracy evaluation
would answer a different question from pipeline efficiency.

## Next

Read [Methodology](methodology.md) for the experimental design and measurement
protocol used by the case studies.
