# SWE-bench Docker resource probe (2026-09-24)

This probe used cached SWE-bench Verified instance images and the dataset's
reference patches. It measured Docker test containers on the team's remote
daemon while the real SWE-bench harness ran. The daemon reported 384 CPUs and
about 1.1 TiB RAM. Images for these tasks were about 2.7–3.1 GB on disk;
image size is not per-container memory use.

| Concurrent test tasks | Harness result | Observed peak combined working memory | Observed peak combined CPU | Elapsed wall time |
| ---: | --- | ---: | ---: | ---: |
| 1 (pytest) | 1/1 completed and resolved | 100 MiB | CPU sample invalid | 53 s |
| 2 (pytest, SymPy) | 2/2 completed and resolved | 181 MiB | ~100% of one CPU core | 101 s |
| 4 (pytest, SymPy, Sphinx, Django) | 3/4 completed and resolved; Django failed on SSH broken pipe | 150 MiB sampled | ~75% of one CPU core sampled | 103 s |

The four-task run reached four live test containers. Its lower sampled peak
than the two-task run reflects sparse SSH polling and one early transport
failure; it is not evidence that four tasks need less memory. Measurements
exclude model-server memory, image storage, and other workloads on the shared
daemon. They also do not establish a safe limit for 28 simultaneous SWE tasks.

The failed Django task started its container and then lost the Docker SSH
connection while copying the patch. Running that same reference patch through
ExGentic's `run_harness()` separately completed and resolved the task. The
Premise path now resolves the SSH-configured port, uses Docker's SSH CLI
transport for harness calls, retries an incomplete harness evaluation once,
and marks persistent transport failures as benchmark errors.

These observations do not support a memory-based limit of two concurrent
SWE sessions. The default `--swebench-sessions 2` remains a conservative
transport setting until a larger load test establishes reliable throughput.
