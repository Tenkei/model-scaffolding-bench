# Viewing stored results

Saved experiment results live under [`results/`](../results/). Each experiment
directory contains one JSON artifact per benchmark configuration, written by
`ml-pipes benchmark --save`.

For the result schema and broader loading and comparison options, see
[ml-pipes benchmarking](https://github.com/trained-by-humans/ml-pipes/blob/main/docs/BENCHMARKING.md).

## Inspect the artifact

The files are ordinary JSON and can be opened directly:

```bash
find results/yolo-cpu-2026-09-14 -name '*.json'
```

Each artifact records its configuration label and metadata, total latency
statistics, and timings for the pipeline operators. Latencies are in
milliseconds.

## Load an artifact in Python

Use `BenchmarkResult.load` to restore a saved result:

```python
from ml_pipes.benchmark import BenchmarkResult

result = BenchmarkResult.load(
    "results/yolo-cpu-2026-09-14/strategy_ultralytics-paths_max_concurrency_1.json"
)

print(result.to_table())
print(result.total.mean_ms)
```

The result is restored as a `BenchmarkResult`, so its total and per-operator
statistics are available for later comparison, tables, or plots without
rerunning the experiment.

## Compare committed experiments

Committed experiment directories are the shared record of past runs. Compare
only like-for-like runs: the same case study, input workload, model, and run
configuration. Check the directory name and the result label before drawing a
conclusion from latency differences.

Use the total timing for end-to-end comparison. Use individual operator timing
to explain where a difference came from, such as source decoding,
preprocessing, or model prediction.

## Next

Return to the [project README](../README.md) to select another case study.
