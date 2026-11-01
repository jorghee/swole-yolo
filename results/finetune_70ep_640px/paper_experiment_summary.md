# Fine-tuning summary (YOLO11n-obb, pretrained on DOTA)

| Dataset | Train | Val | Test | Test mAP50 | Test mAP50-95 | Precision | Recall | Latency ms/img CPU | FPS CPU | Train s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2500 | 1748 | 376 | 376 | 0.9394 | 0.8175 | 0.9281 | 0.8595 | 33.78 | 29.60 | 992 |

Test split evaluated once with the best checkpoint; clip-level splits shared with the from-scratch baseline.
CPU latency measured with limited threads as a Raspberry Pi proxy; final numbers must be measured on-device (NCNN export).
