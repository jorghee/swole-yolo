# Fine-tuning summary (YOLO11n-obb, pretrained on DOTA)

| Dataset | Train | Val | Test | Test mAP50 | Test mAP50-95 | Precision | Recall | Latency ms/img CPU | FPS CPU | Train s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 700 | 150 | 150 | 0.6622 | 0.5483 | 0.7742 | 0.6425 | 30.99 | 32.27 | 242 |
| 1500 | 1050 | 225 | 225 | 0.8167 | 0.6769 | 0.8352 | 0.7058 | 32.11 | 31.14 | 340 |
| 2000 | 1400 | 300 | 300 | 0.8129 | 0.6873 | 0.8616 | 0.7219 | 32.98 | 30.32 | 444 |
| 2500 | 1748 | 376 | 376 | 0.9140 | 0.7847 | 0.9053 | 0.8517 | 32.89 | 30.41 | 547 |
| 3000 | 2098 | 450 | 452 | 0.8848 | 0.7546 | 0.8540 | 0.8333 | 33.10 | 30.21 | 652 |

Test split evaluated once with the best checkpoint; clip-level splits shared with the from-scratch baseline.
CPU latency measured with limited threads as a Raspberry Pi proxy; final numbers must be measured on-device (NCNN export).
