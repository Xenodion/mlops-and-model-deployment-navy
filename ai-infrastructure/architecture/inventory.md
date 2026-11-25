# Vertex release inventory

| Local Parquet input | File bytes | Sensor readings |
|---|---:|---:|
| GBG500 raw | 30,206,760 | 1,529,547 |
| GBG500 training split | 24,908,635 | 1,224,731 |
| GBG500 golden split | 6,592,372 | 304,815 |
| SRV3490 raw | 268,229,940 | 16,892,002 |
| SRV3490 cleaned | 226,115,704 | 16,891,993 |

File lengths and Parquet row counts were read on November 24. Raw and processed rows describe overlapping data. The replay uses the 100 golden rides; its prepared classifier tensor has shape (1,885 windows, 6 channels, 300 readings). The stored fall training evaluation reports 30,293 training windows and 205 independent held-out comparison groups.

| Measured on the downloaded production artifacts | Ride classifier v2 | Fall risk v1 |
|---|---:|---:|
| Serialized MLflow Python model (bytes) | 193,215 | 9,197 |
| Complete MLflow model directory (bytes) | 1,219,851 | 985,226 |

The directory totals sum file lengths recursively and include code and environment specifications. Both models run inside the same API process. The trained artifact sizes answer the storage question. Peak process RSS is needed to size the worker.

The [cloud replay](../benchmarks/results/vertex-c193360-cpu-replay.json) records the image digest, both source runs and a 100-ride correctness check. Each ride contributes one complete-ride response and two consecutive live responses. The fall-risk checks validate the response structure and score range. Measuring discrimination requires the separate labeled SRV3490 held-out groups.
