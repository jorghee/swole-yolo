# Raspberry Pi benchmark (YOLO11n-obb, NCNN, batch=1)

Device: aarch64 / Linux-6.18.34+rpt-rpi-v8-aarch64-with-glibc2.41

| Model | imgsz | Latency ms (mean +/- std) | p95 ms | FPS | .pt MB | NCNN MB | CPU % mean | CPU % max | T start->end (C) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| m1_1000img_544 | 544 | 302.4 +/- 5.3 | 311.5 | 3.31 | - | 10.23 | 81.7 | 84.2 | 48.199 -> 65.244 |
| m2_2500img_544 | 544 | 300.5 +/- 2.6 | 305.3 | 3.33 | - | 10.23 | 82.6 | 84.1 | 50.634 -> 70.114 |
| m3_2500img_640 | 640 | 399.9 +/- 3.6 | 405.9 | 2.50 | - | 10.25 | 84.1 | 85.5 | 49.66 -> 70.114 |

Latency is end-to-end per image (preprocess + network + NMS), batch=1, measured over the standardized
test bank after warmup. throttled_flag=0x0 means the Pi never throttled during measurement.
