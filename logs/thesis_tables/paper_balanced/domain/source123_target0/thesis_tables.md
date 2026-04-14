# Overnight Run Tables

## Table 0 - Preprocess x Model Matrix

| Preprocess | Model | Shot | Metric Protocol | Accuracy (%) | Latency (ms) |
| --- | --- | --- | --- | --- | --- |
| FFT | CNN | 5 | deployment_baseline | 96.35% | 0.9705 |
| FFT | CNN | 10 | deployment_baseline | 87.5% | 1.2666 |
| FFT | CNN | 15 | deployment_baseline | 92.84% | 1.1872 |
| FFT | MAML | 5 | deployment_baseline | 97.47% | 1.2303 |
| FFT | MAML | 10 | deployment_baseline | 94.52% | 1.1513 |
| FFT | MAML | 15 | deployment_baseline | 100.0% | 1.1018 |
| FFT | ProtoNet | 5 | deployment_baseline | 99.86% | 1.2587 |
| FFT | ProtoNet | 10 | deployment_baseline | 99.86% | 1.1942 |
| FFT | ProtoNet | 15 | deployment_baseline | 100.0% | 1.2404 |
| STFT | CNN | 5 | deployment_baseline | 81.6% | 2.718 |
| STFT | CNN | 10 | deployment_baseline | 93.68% | 2.8804 |
| STFT | CNN | 15 | deployment_baseline | 93.54% | 2.7157 |
| STFT | MAML | 5 | deployment_baseline | 96.77% | 2.889 |
| STFT | MAML | 10 | deployment_baseline | 97.33% | 2.6667 |
| STFT | MAML | 15 | deployment_baseline | 96.35% | 2.8577 |
| STFT | ProtoNet | 5 | deployment_baseline | 98.17% | 2.8 |
| STFT | ProtoNet | 10 | deployment_baseline | 97.33% | 2.7818 |
| STFT | ProtoNet | 15 | deployment_baseline | 98.74% | 2.6528 |
| WT | CNN | 5 | deployment_baseline | 86.1% | 2.7304 |
| WT | CNN | 10 | deployment_baseline | 92.28% | 2.7657 |
| WT | CNN | 15 | deployment_baseline | 91.57% | 2.745 |
| WT | MAML | 5 | deployment_baseline | 93.4% | 2.873 |
| WT | MAML | 10 | deployment_baseline | 97.89% | 3.0974 |
| WT | MAML | 15 | deployment_baseline | 95.08% | 2.7979 |
| WT | ProtoNet | 5 | deployment_baseline | 99.02% | 2.8875 |
| WT | ProtoNet | 10 | deployment_baseline | 99.3% | 2.6654 |
| WT | ProtoNet | 15 | deployment_baseline | 99.44% | 2.868 |

## Locked Thesis Tables

# Thesis Tables

## Locked Profile

| dataset | preprocess | ways | train_domains | test_domain | runtime_backend | enable_compression | prune_ratio | fault_labels |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CWRU | STFT | 5 | 0,1,2 | 3 | onnxruntime | True | 0.4 |  |

## Table 1 - Model Performance

No data available.

## Table 2 - Few-Shot Performance

No data available.

## Table 3 - Compression Impact

No data available.

## Table 4 - System Performance

No data available.
