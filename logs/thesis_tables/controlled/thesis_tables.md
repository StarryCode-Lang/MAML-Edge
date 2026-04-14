# Overnight Run Tables

## Table 0 - Preprocess x Model Matrix

| Preprocess | Model | Shot | Metric Protocol | Accuracy (%) | Latency (ms) |
| --- | --- | --- | --- | --- | --- |
| FFT | CNN | 5 | deployment_baseline | 85.07% | 1.2193 |
| FFT | CNN | 10 | deployment_baseline | 91.48% | 1.0674 |
| FFT | CNN | 15 | deployment_baseline | 91.9% | 1.1826 |
| FFT | MAML | 5 | deployment_baseline | 87.49% | 1.2277 |
| FFT | MAML | 10 | deployment_baseline | 99.68% | 1.2684 |
| FFT | MAML | 15 | deployment_baseline | 99.79% | 1.2062 |
| FFT | ProtoNet | 5 | deployment_baseline | 100.0% | 1.2038 |
| FFT | ProtoNet | 10 | deployment_baseline | 100.0% | 1.2217 |
| FFT | ProtoNet | 15 | deployment_baseline | 100.0% | 1.2075 |
| STFT | CNN | 5 | deployment_baseline | 99.05% | 2.7751 |
| STFT | CNN | 10 | deployment_baseline | 98.32% | 2.7819 |
| STFT | CNN | 15 | deployment_baseline | 98.32% | 2.7181 |
| STFT | MAML | 5 | deployment_baseline | 99.16% | 2.899 |
| STFT | MAML | 10 | deployment_baseline | 99.26% | 2.7164 |
| STFT | MAML | 15 | deployment_baseline | 99.47% | 2.7156 |
| STFT | ProtoNet | 5 | deployment_baseline | 100.0% | 2.7264 |
| STFT | ProtoNet | 10 | deployment_baseline | 100.0% | 2.7676 |
| STFT | ProtoNet | 15 | deployment_baseline | 100.0% | 2.729 |
| WT | CNN | 5 | deployment_baseline | 99.68% | 2.7852 |
| WT | CNN | 10 | deployment_baseline | 99.89% | 2.6192 |
| WT | CNN | 15 | deployment_baseline | 99.79% | 2.6741 |
| WT | MAML | 5 | deployment_baseline | 99.26% | 2.5968 |
| WT | MAML | 10 | deployment_baseline | 99.37% | 2.7706 |
| WT | MAML | 15 | deployment_baseline | 100.0% | 2.673 |
| WT | ProtoNet | 5 | deployment_baseline | 99.79% | 2.6308 |
| WT | ProtoNet | 10 | deployment_baseline | 100.0% | 2.6244 |
| WT | ProtoNet | 15 | deployment_baseline | 99.89% | 2.4148 |

## Locked Thesis Tables

# Thesis Tables

## Locked Profile

| dataset | preprocess | ways | train_domains | test_domain | runtime_backend | enable_compression | prune_ratio | fault_labels |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CWRU | STFT | 5 | 0,1,2 | 3 | onnxruntime | True | 0.4 |  |

## Table 1 - Model Performance

| Model | Metric Protocol | Accuracy (%) |
| --- | --- | --- |
| CNN | deployment_baseline | 99.05% |
| MAML | deployment_baseline | 99.16% |
| ProtoNet | deployment_baseline | 100.0% |

## Table 2 - Few-Shot Performance

| Model | Shot | Metric Protocol | Accuracy (%) |
| --- | --- | --- | --- |
| MAML | 5 | deployment_baseline | 99.16% |
| MAML | 10 | deployment_baseline | 99.26% |
| MAML | 15 | deployment_baseline | 99.47% |

## Table 3 - Compression Impact

| Variant | Metric Protocol | Accuracy (%) | Latency (ms) | Parameter Count | Model Size (MB) |
| --- | --- | --- | --- | --- | --- |
| Original MAML | deployment_baseline | 99.16% | 2.899 | 113738 | 0.436068 |
| Pruned | deployment_float | 99.26% | 2.8078 | 40860 | 0.158854 |
| Pruned + INT8 | deployment_int8 | 99.26% | 3.2238 | 40860 | 0.056938 |

## Table 4 - System Performance

No data available.
