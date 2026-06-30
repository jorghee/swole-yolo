# Dataset audit summary

- Images in CSV: 300
- Images on disk: 300
- Missing images: 0
- Unique clips: 300
- Empty frames: 17
- Total annotated objects: 3240
- Out-of-bounds objects: 0

## Class distribution

| Class | Objects | Percentage |
| --- | ---: | ---: |
| car | 2610 | 80.56% |
| van | 57 | 1.76% |
| microbus | 19 | 0.59% |
| minibus | 94 | 2.90% |
| bus | 14 | 0.43% |
| articulated_bus | 3 | 0.09% |
| truck | 163 | 5.03% |
| mototaxi | 33 | 1.02% |
| motorcycle | 247 | 7.62% |

## Paper-useful claims to verify

- Dataset integrity can be reported using `dataset_summary.json`.
- Class imbalance can motivate Macro AP and class-aware sampling.
- Clip-level splitting should be used in all experiments.
