# Running experiments

See the [Case-study index](index.md) to select a case study and find its
install command, pipeline variants, supported configuration, and example run.

> [!NOTE]
> This guide covers the repository workflow. For the complete CLI and benchmark
axis reference, see [ml-pipes benchmarking](https://github.com/trained-by-humans/ml-pipes/blob/main/docs/BENCHMARKING.md).

## 1. Prepare the execution provider

Prepare the machine that will run the experiment before installing its
case-study dependencies. Ensure the repository and framework cache paths are
on persistent, writable storage so downloaded models and example inputs can be
reused between sessions.

> [!TIP]
> See [AWS EC2](aws.md) for the CUDA workflow provided by AWS.
For example, use its Deep Learning AMI setup and confirm the selected GPU with
`nvidia-smi` before running the Ultralytics case study.

## 2. Set up the case-study environment

Create and activate an environment named after the selected case study.
For example, for Ultralytics YOLO:

```bash
python3 -m venv .venv-ultralytics
source .venv-ultralytics/bin/activate
python -m pip install --upgrade pip
```

> [!CAUTION]
> Do not install all case-study extras into one environment. The frameworks and
> their GPU-runtime dependencies can require incompatible package versions.
> Create and use a separate environment for each case study you run.


Install the extra listed for the selected case study in the
[Case-study index](index.md). Run an unmeasured invocation first when a
model or example input must be downloaded or prepared.

For example, install the Ultralytics case study with:

```bash
python -m pip install -e '.[ultralytics]'
```

## 3. Run and compare a case study

Use the example command in the [Case-study index](index.md), adjusting
its supported axes for the desired workload. The terminal table reports
end-to-end latency and an operator-level timing breakdown for every
configuration.

Run the explicit strategy once with `max_concurrency=1` for its **serial** route,
then again with a value greater than one for its **parallel** route.
`max_concurrency` sets external `Scatter` workers and, whenever the model
scaffolding exposes it, the corresponding native workers. Compare total
latency, then use the `Scatter` child timings to identify the preparation
effect.

For example, compare Ultralytics' native path route with serial and concurrent
explicit decoding:

```bash
python -m ml_pipes benchmark src.run_batch_decode_comparison \
  --axis strategy=ultralytics-paths,scatter-ultralytics-decode,scatter-decode \
  --axis max_concurrency=1,8 \
  --data-axis batch_size=8 \
  --runs 20 --warmup 3
```

## 4. Save the result and environment

Append `--save results/<experiment-name>` to an index command. Choose a concise
name that identifies the case study, environment, workload, and date, such as
`yolo-cpu-macos-b8-2026-09-14`.

For example, save the same Ultralytics comparison with:

```bash
python -m ml_pipes benchmark src.run_batch_decode_comparison \
  --axis strategy=ultralytics-paths,scatter-ultralytics-decode,scatter-decode \
  --axis max_concurrency=1,8 \
  --data-axis batch_size=8 \
  --runs 20 --warmup 3 \
  --save results/yolo-cpu-macos-b8-2026-09-15
```

`ml-pipes` writes one JSON file per active configuration. Treat the directory
as immutable: make a new directory for a rerun, changed command, model,
hardware, or environment.

After saving a result, capture the active environment in the same directory:

```bash
python scripts/capture_environment.py results/<experiment-name>
```

The script writes `environment.txt` containing `nvidia-smi`, `python --version`,
and `python -m pip freeze`. It does not overwrite an existing capture.

## Next

Read [Viewing stored results](viewing-results.md) to inspect and restore the
saved benchmark artifacts.
