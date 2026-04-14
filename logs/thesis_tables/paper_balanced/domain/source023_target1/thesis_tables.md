# Overnight Run Tables

## Table 0 - Preprocess x Model Matrix

| Preprocess | Model | Shot | Metric Protocol | Accuracy (%) | Latency (ms) |
| --- | --- | --- | --- | --- | --- |
| FFT | CNN | 5 | deployment_baseline | 89.75% | 1.4874 |
| FFT | CNN | 10 | deployment_baseline | 87.74% | 1.0958 |
| FFT | CNN | 15 | deployment_baseline | 87.95% | 1.3935 |
| FFT | MAML | 5 | deployment_baseline | 99.37% | 1.4332 |
| FFT | MAML | 10 | deployment_baseline | 99.37% | 1.3931 |
| FFT | MAML | 15 | deployment_baseline | 99.15% | 1.4848 |
| FFT | ProtoNet | 5 | deployment_baseline | 100.0% | 1.3136 |
| FFT | ProtoNet | 10 | deployment_baseline | 99.37% | 1.3473 |
| FFT | ProtoNet | 15 | deployment_baseline | 100.0% | 1.1786 |
| STFT | CNN | 5 | deployment_baseline | 96.09% | 2.6819 |
| STFT | CNN | 10 | deployment_baseline | 95.35% | 2.7413 |
| STFT | CNN | 15 | deployment_baseline | 98.41% | 2.2351 |
| STFT | MAML | 5 | deployment_baseline | 99.15% | 2.8632 |
| STFT | MAML | 10 | deployment_baseline | 99.58% | 2.7204 |
| STFT | MAML | 15 | deployment_baseline | 99.79% | 2.9149 |
| STFT | ProtoNet | 5 | deployment_baseline | 99.79% | 2.8486 |
| STFT | ProtoNet | 10 | deployment_baseline | 99.68% | 2.8373 |
| STFT | ProtoNet | 15 | deployment_baseline | 100.0% | 2.7441 |
| WT | CNN | 5 | deployment_baseline | 91.54% | 2.7393 |
| WT | CNN | 10 | deployment_baseline | 97.99% | 2.628 |
| WT | CNN | 15 | deployment_baseline | 96.72% | 2.5978 |
| WT | MAML | 5 | deployment_baseline | 99.26% | 2.6828 |
| WT | MAML | 10 | deployment_baseline | 99.58% | 2.7169 |
| WT | MAML | 15 | deployment_baseline | 98.63% | 2.9436 |
| WT | ProtoNet | 5 | deployment_baseline | 98.84% | 2.7634 |
| WT | ProtoNet | 10 | deployment_baseline | 99.37% | 2.7309 |
| WT | ProtoNet | 15 | deployment_baseline | 99.05% | 2.562 |

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
