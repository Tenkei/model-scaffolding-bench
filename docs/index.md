# Case-study index

This catalog helps you select a case study. Each study compares a model
provider's native input path with pipelines that make loading, decoding,
preprocessing, batching, or scheduling explicit.

| Case study          | Model / Task                                      | Install                                      | Module                                             | Default input                               |
|---------------------|---------------------------------------------------|----------------------------------------------|----------------------------------------------------|---------------------------------------------|
| Ultralytics YOLO    | `yolo26n.pt` <br> object detection                | `python -m pip install -e '.[ultralytics]'`  | `src.run_batch_decode_comparison`                  | `assets/images/ultralytics-sample.jpg`      |
| PaddleOCR detection | `PP-OCRv6_tiny_det` <br> text detection           | `python -m pip install -e '.[paddleocr]'`    | `src.run_paddleocr_batch_decode_comparison`        | Cached 4K PaddleOCR sample                  |
| Hugging Face ViT    | `google/vit-base-patch16-224` <br> classification | `python -m pip install -e '.[transformers]'` | `src.run_transformers_vit_batch_decode_comparison` | Cached 4K version of the Ultralytics sample |

PaddleOCR also requires a [PaddlePaddle runtime](https://www.paddlepaddle.org.cn/install/quick).
Install `paddlepaddle` for CPU runs, or the wheel compatible with the
selected CUDA environment for GPU runs.

> [!NOTE]
> Saved measurements are documented separately in
> [Viewing stored results](viewing-results.md).

> [!TIP]
> Run all commands from the repository root after preparing the virtual environment in
> [Running experiments](running-experiments.md).

## Common parameters

Each command invokes `python -m ml_pipes benchmark <module>`, which loads the
case study's pipeline and data factories and runs every selected configuration.

| Component | Configuration | Role |
| --- | --- | --- |
| Pipeline | `--axis ...` | Builds the inference pipeline, including its strategy, device, and concurrency settings. |
| Data | `--data-axis ...` | Builds the inputs supplied to each pipeline call. |
| Benchmark | `--runs`, `--warmup`, and `--save` | Controls measurement collection, warmup calls, and result persistence. |

Every case study accepts the parameters below. The available `strategy` values
are listed with each case study. For the complete CLI and benchmark-axis
reference, see [ml-pipes benchmarking](https://github.com/trained-by-humans/ml-pipes/blob/main/docs/BENCHMARKING.md).

| Category | Parameter | Purpose |
| --- | --- | --- |
| Pipeline | `--axis strategy=...` | Selects the pipeline variants to compare. |
| Pipeline | `--axis device=...` | Selects the framework device, such as `cpu`, `cuda`, `cuda:0`, or `mps`, when supported by the installed framework. |
| Pipeline | `--axis max_concurrency=...` | Sets the number of workers used by explicit `Scatter` stages and, when exposed by the native model scaffolding, its corresponding workers. |
| Data | `--data-axis batch_size=...` | Sets the number of inputs passed to one pipeline call. |
| Data | `--data-axis source=...` | Replaces the case study's default input with a local file. |
| Benchmark | `--runs N` | Collects `N` timed measurements per configuration. |
| Benchmark | `--warmup N` | Runs `N` unmeasured calls before collection. |
| Benchmark | `--save results/<experiment-name>` | Saves one JSON artifact per active configuration. |

## Ultralytics YOLO detection

| Strategy | Pipeline boundary under test |
| --- | --- |
| `ultralytics-paths` | Native Ultralytics path loading and prediction. |
| `scatter-ultralytics-decode` | Concurrently run Ultralytics' path-to-array loader, then predict. |
| `scatter-decode` | Concurrently load and decode files, convert to YOLO's BGR array boundary, then predict. |

Additional pipeline axes: `model`, `conf`, and `imgsz`.

```bash
python -m ml_pipes benchmark src.run_batch_decode_comparison \
  --axis strategy=ultralytics-paths,scatter-ultralytics-decode,scatter-decode \
  --axis max_concurrency=1,8 \
  --data-axis batch_size=1,4,8 \
  --runs 20 --warmup 3
```

## PaddleOCR text detection

| Strategy | Pipeline boundary under test |
| --- | --- |
| `paddleocr-paths` | Native `TextDetection.predict` path loading and prediction. |
| `scatter-paddleocr-decode` | Concurrently run PaddleOCR's BGR reader, then predict. |
| `scatter-decode` | Concurrently load and decode files, convert to PaddleOCR's BGR array boundary, then predict. |

Additional pipeline axis: `model_name`. The default model is
`PP-OCRv6_tiny_det`, chosen to keep input-stage cost measurable.

```bash
python -m ml_pipes benchmark src.run_paddleocr_batch_decode_comparison \
  --axis strategy=paddleocr-paths,scatter-paddleocr-decode,scatter-decode \
  --axis max_concurrency=1,8 \
  --data-axis batch_size=1,4,8 \
  --runs 20 --warmup 3
```

## Hugging Face ViT image classification

| Strategy | Pipeline boundary under test |
| --- | --- |
| `transformers-paths` | Native `transformers.pipeline` path loading, preprocessing, prediction, and postprocessing. |
| `scatter-transformers-decode` | Concurrently run Transformers' image loader, then use its native pipeline. |
| `direct-model` | Decode inputs and batch preprocessing explicitly, then invoke the same ViT model directly. |
| `direct-model-concurrent-preprocess` | Concurrently decode and preprocess inputs before batching and direct model invocation. |

Additional pipeline axes: `model_name` and `inference_batch_size`.
`inference_batch_size` is the model-facing batch size; `batch_size` is the
number of inputs passed to the pipeline.

```bash
python -m ml_pipes benchmark src.run_transformers_vit_batch_decode_comparison \
  --axis strategy=transformers-paths,scatter-transformers-decode,direct-model,direct-model-concurrent-preprocess \
  --axis inference_batch_size=1,8 \
  --axis max_concurrency=1,8 \
  --data-axis batch_size=1,8 \
  --runs 20 --warmup 3
```
