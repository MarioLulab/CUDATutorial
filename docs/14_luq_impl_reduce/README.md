| 优化手段 | 运行时间(us) | 带宽| 加速比 |
| --- | --- | --- | --- |
| Naive | 133.22 | 29.435 GB/s | ~ |
| Interleaved | 77.31 | 50.727 GB/s | 1.723 |    bank-conflict
| Sequential Addressing | 62.062 | 61.534 GB/s | 2.147 |    bank-free
| Unroll First | 40.656 | 95.13 GB/s | 3.277 | unroll_num = 2
| Unroll First | 25.538 | 151.65 GB/s | 5.217 | unroll_num = 4
| Unroll First and Unroll last | 32.736 | 117.61 GB/s | 4.070 | unroll_num = 2
| Unroll First and Unroll last | 21.610 | 178.21 GB/s | 6.165 | unroll_num = 4
| Unroll First and Unroll all | 33.755 | 115.26 GB/s | 3.947 | unroll_num = 2
| Unroll First and Unroll all | 22.018 | 174.94 GB/s | 6.050 | unroll_num = 4