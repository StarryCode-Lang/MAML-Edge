# Overnight Run Tables

## Table 0 - Preprocess x Model Matrix

| Preprocess | Model | Shot | Metric Protocol | Accuracy (%) | Latency (ms) |
| --- | --- | --- | --- | --- | --- |
| FFT | CNN | 5 | deployment_baseline | 53.03% | 1.2356 |
| FFT | CNN | 10 | deployment_baseline | 61.45% | 1.2599 |
| FFT | CNN | 15 | deployment_baseline | 29.29% | 0.9282 |
| FFT | MAML | 5 | deployment_baseline | 81.82% | 1.183 |
| FFT | MAML | 10 | deployment_baseline | 99.83% | 1.2229 |
| FFT | MAML | 15 | deployment_baseline | 66.16% | 1.0958 |
| FFT | ProtoNet | 5 | deployment_baseline | 99.83% | 0.931 |
| FFT | ProtoNet | 10 | deployment_baseline | 100.0% | 1.1955 |
| FFT | ProtoNet | 15 | deployment_baseline | 100.0% | 1.1208 |
| STFT | CNN | 5 | deployment_baseline | 74.92% | 2.52 |
| STFT | CNN | 10 | deployment_baseline | 73.74% | 2.7839 |
| STFT | CNN | 15 | deployment_baseline | 85.02% | 3.0984 |
| STFT | MAML | 5 | deployment_baseline | 98.48% | 2.9263 |
| STFT | MAML | 10 | deployment_baseline | 99.66% | 2.9247 |
| STFT | MAML | 15 | deployment_baseline | 100.0% | 2.9543 |
| STFT | ProtoNet | 5 | deployment_baseline | 99.16% | 2.69 |
| STFT | ProtoNet | 10 | deployment_baseline | 99.33% | 2.6546 |
| STFT | ProtoNet | 15 | deployment_baseline | 99.16% | 2.6905 |
| WT | CNN | 5 | deployment_baseline | 95.12% | 2.5512 |
| WT | CNN | 10 | deployment_baseline | 87.88% | 2.7005 |
| WT | CNN | 15 | deployment_baseline | 93.27% | 2.9358 |
| WT | MAML | 5 | deployment_baseline | 99.33% | 2.7024 |
| WT | MAML | 10 | deployment_baseline | 99.83% | 2.4183 |
| WT | MAML | 15 | deployment_baseline | 99.83% | 2.6298 |
| WT | ProtoNet | 5 | deployment_baseline | 99.66% | 2.7041 |
| WT | ProtoNet | 10 | deployment_baseline | 99.83% | 2.6784 |
| WT | ProtoNet | 15 | deployment_baseline | 99.83% | 2.6678 |

## Locked Thesis Tables

# Thesis Tables

## Locked Profile

| dataset | preprocess | ways | train_domains | test_domain | runtime_backend | enable_compression | prune_ratio | fault_labels |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CWRU | STFT | 5 | 0,1,2 | 3 | onnxruntime | True | 0.4 |  |

## Table 1 - Model Performance

| Model | Metric Protocol | Accuracy (%) |
| --- | --- | --- |
| CNN | deployment_baseline | 74.92% |
| MAML | deployment_baseline | 98.48% |
| ProtoNet | deployment_baseline | 99.16% |

## Table 2 - Few-Shot Performance

| Model | Shot | Metric Protocol | Accuracy (%) |
| --- | --- | --- | --- |
| MAML | 5 | deployment_baseline | 98.48% |
| MAML | 10 | deployment_baseline | 99.66% |
| MAML | 15 | deployment_baseline | 100.0% |

## Table 3 - Compression Impact

| Variant | Metric Protocol | Accuracy (%) | Latency (ms) | Parameter Count | Model Size (MB) |
| --- | --- | --- | --- | --- | --- |
| Original MAML | deployment_baseline | 98.48% | 2.9263 | 113738 | 0.436068 |
| Pruned | deployment_float | 98.99% | 2.8617 | 40860 | 0.158854 |
| Pruned + INT8 | deployment_int8 | 98.82% | 3.455 | 40860 | 0.056937 |

## Table 4 - System Performance

No data available.
