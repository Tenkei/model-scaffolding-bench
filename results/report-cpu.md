# CPU benchmark results

These results were collected on CPU with Python 3.12.10. Each comparison uses an input batch of eight images and `max_concurrency=8`; the ViT experiment also uses `inference_batch_size=8`. Each configuration contains 20 measured calls.

See [AWS GPU benchmark results](report-aws-g6.x2large.md) for the corresponding GPU runs.

## Summary

| Experiment model | Native paths (mean) | Explicit pipeline (mean) | Difference |
| --- | ---: | ---: | ---: |
| Ultralytics YOLO detection | 228.80 ms | 196.77 ms | 14.0% lower |
| PaddleOCR text detection | 1747.85 ms | 1619.59 ms | 7.3% lower |
| ViT image classification | 1307.04 ms | 724.23 ms | 44.6% lower |

“Native paths” is the package's path-based route. “Explicit pipeline” is the fully explicit decoding/preprocessing route, using concurrent preprocessing for ViT.

## Ultralytics YOLO detection

Source: [`ultralytics-2026-09-15`](ultralytics-2026-09-15/)

| Strategy | Mean total | P95 total | Gain over native-path baseline |
| --- | ---: | ---: | ---: |
| Ultralytics native paths | 228.80 ms | 250.76 ms | — |
| Scatter + Ultralytics decode | 205.92 ms | 220.60 ms | 10.0% lower |
| Fully explicit decode | 196.77 ms | 203.97 ms | 14.0% lower |

## PaddleOCR text detection

Source: [`paddleocr-2026-09-15`](paddleocr-2026-09-15/)

| Strategy | Mean total | P95 total | Gain over native-path baseline |
| --- | ---: | ---: | ---: |
| PaddleOCR native paths | 1747.85 ms | 1753.70 ms | — |
| Scatter + PaddleOCR decode | 1615.64 ms | 1643.22 ms | 7.6% lower |
| Fully explicit decode | 1619.59 ms | 1646.33 ms | 7.3% lower |

## ViT image classification

Source: [`vit-2026-09-15`](vit-2026-09-15/)

| Strategy | Mean total | P95 total | Gain over native-path baseline |
| --- | ---: | ---: | ---: |
| Transformers native paths | 1307.04 ms | 1327.59 ms | — |
| Scatter + Transformers decode | 1074.16 ms | 1136.03 ms | 17.8% lower |
| Direct model + batched preprocess | 939.60 ms | 987.37 ms | 28.1% lower |
| Direct model + concurrent explicit preprocess | 724.23 ms | 748.61 ms | 44.6% lower |
