# AWS GPU benchmark results

These results were collected on an AWS `g6.2xlarge` instance with Python 3.12.10. Each comparison uses an input batch of eight images and `max_concurrency=8`; the ViT experiment also uses `inference_batch_size=8`.

See [CPU benchmark results](report-cpu.md) for the corresponding CPU runs.

## Summary

| Experiment model | Native paths (mean) | Explicit pipeline (mean) | Difference |
| --- | ---: | ---: | ---: |
| Ultralytics YOLO detection | 65.95 ms | 32.37 ms | 50.9% lower |
| PaddleOCR text detection | 374.25 ms | 198.44 ms | 47.0% lower |
| ViT image classification | 1786.22 ms | 593.35 ms | 66.8% lower |

“Native paths” is the package's path-based route. “Explicit pipeline” is the fully explicit decoding/preprocessing route, using concurrent preprocessing for ViT.

## Ultralytics YOLO detection

Source: [`ultralytics-g6.2xlarge-py3.12.10-2026-09-16`](ultralytics-g6.2xlarge-py3.12.10-2026-09-16/)

| Strategy | Mean total | P95 total | Gain over native-path baseline |
| --- | ---: | ---: | ---: |
| Ultralytics native paths | 65.95 ms | 71.93 ms | — |
| Scatter + Ultralytics decode | 37.99 ms | 38.57 ms | 42.4% lower |
| Fully explicit decode | 32.37 ms | 32.85 ms | 50.9% lower |

## PaddleOCR text detection

Source: [`paddleocr-g6.2xlarge-py3.12.10-2026-09-16`](paddleocr-g6.2xlarge-py3.12.10-2026-09-16/)

| Strategy | Mean total | P95 total | Gain over native-path baseline |
| --- | ---: | ---: | ---: |
| PaddleOCR native paths | 374.25 ms | 383.50 ms | — |
| Scatter + PaddleOCR decode | 196.90 ms | 197.94 ms | 47.4% lower |
| Fully explicit decode | 198.44 ms | 200.25 ms | 47.0% lower |

## ViT image classification

Source: [`vit-g6.2xlarge-py3.12.10-2026-09-16`](vit-g6.2xlarge-py3.12.10-2026-09-16/)

| Strategy | Mean total | P95 total | Gain over native-path baseline |
| --- | ---: | ---: | ---: |
| Transformers native paths | 1786.22 ms | 1819.19 ms | — |
| Scatter + Transformers decode | 1540.85 ms | 1577.28 ms | 13.7% lower |
| Direct model + batched preprocess | 890.12 ms | 898.53 ms | 50.2% lower |
| Direct model + concurrent explicit preprocess | 593.35 ms | 606.20 ms | 66.8% lower |
