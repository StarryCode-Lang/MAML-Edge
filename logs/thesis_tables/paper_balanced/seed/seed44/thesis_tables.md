# Overnight Run Tables

## Table 0 - Preprocess x Model Matrix

| Preprocess | Model | Shot | Metric Protocol | Accuracy (%) | Latency (ms) |
| --- | --- | --- | --- | --- | --- |
| FFT | CNN | 5 | deployment_baseline | 68.07% | 1.0205 |
| FFT | CNN | 10 | deployment_baseline | 69.13% | 1.1755 |
| FFT | CNN | 15 | deployment_baseline | 79.45% | 1.0768 |
| FFT | MAML | 5 | deployment_baseline | 100.0% | 0.9742 |
| FFT | MAML | 10 | deployment_baseline | 100.0% | 1.1552 |
| FFT | MAML | 15 | deployment_baseline | 99.89% | 1.2609 |
| FFT | ProtoNet | 5 | deployment_baseline | 100.0% | 0.982 |
| FFT | ProtoNet | 10 | deployment_baseline | 100.0% | 1.2526 |
| FFT | ProtoNet | 15 | deployment_baseline | 100.0% | 1.3603 |
| STFT | CNN | 5 | deployment_baseline | 72.5% | 2.9188 |
| STFT | CNN | 10 | deployment_baseline | 85.56% | 2.5995 |
| STFT | CNN | 15 | deployment_baseline | 90.73% | 2.7107 |
| STFT | MAML | 5 | deployment_baseline | 100.0% | 2.9439 |
| STFT | MAML | 10 | deployment_baseline | 100.0% | 2.7964 |
| STFT | MAML | 15 | deployment_baseline | 97.05% | 2.8698 |
| STFT | ProtoNet | 5 | deployment_baseline | 100.0% | 2.7758 |
| STFT | ProtoNet | 10 | deployment_baseline | 100.0% | 2.9433 |
| STFT | ProtoNet | 15 | deployment_baseline | 100.0% | 2.7388 |
| WT | CNN | 5 | deployment_baseline | 97.58% | 2.6214 |
| WT | CNN | 10 | deployment_baseline | 98.21% | 2.7026 |
| WT | CNN | 15 | deployment_baseline | 99.05% | 2.6659 |
| WT | MAML | 5 | deployment_baseline | 100.0% | 2.3937 |
| WT | MAML | 10 | deployment_baseline | 99.16% | 2.8691 |
| WT | MAML | 15 | deployment_baseline | 99.89% | 2.7081 |
| WT | ProtoNet | 5 | deployment_baseline | 99.89% | 2.4347 |
| WT | ProtoNet | 10 | deployment_baseline | 99.89% | 2.7152 |
| WT | ProtoNet | 15 | deployment_baseline | 99.89% | 2.6828 |

## Locked Thesis Tables

# Thesis Tables

## Locked Profile

| dataset | preprocess | ways | train_domains | test_domain | runtime_backend | enable_compression | prune_ratio | fault_labels |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CWRU | STFT | 5 | 0,1,2 | 3 | onnxruntime | True | 0.4 |  |

## Table 1 - Model Performance

| Model | Metric Protocol | Accuracy (%) |
| --- | --- | --- |
| CNN | deployment_baseline | 72.5% |
| MAML | deployment_baseline | 100.0% |
| ProtoNet | deployment_baseline | 100.0% |

## Table 2 - Few-Shot Performance

| Model | Shot | Metric Protocol | Accuracy (%) |
| --- | --- | --- | --- |
| MAML | 5 | deployment_baseline | 100.0% |
| MAML | 10 | deployment_baseline | 100.0% |
| MAML | 15 | deployment_baseline | 97.05% |

## Table 3 - Compression Impact

| Variant | Metric Protocol | Accuracy (%) | Latency (ms) | Parameter Count | Model Size (MB) |
| --- | --- | --- | --- | --- | --- |
| Original MAML | deployment_baseline | 100.0% | 2.9439 | 113738 | 0.436068 |
| Pruned | deployment_float | 99.89% | 2.8004 | 40860 | 0.158854 |
| Pruned + INT8 | deployment_int8 | 99.89% | 3.3458 | 40860 | 0.056938 |

## Table 4 - System Performance

No data available.
