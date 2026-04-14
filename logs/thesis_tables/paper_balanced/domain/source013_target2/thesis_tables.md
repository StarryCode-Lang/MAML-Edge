# Overnight Run Tables

## Table 0 - Preprocess x Model Matrix

| Preprocess | Model | Shot | Metric Protocol | Accuracy (%) | Latency (ms) |
| --- | --- | --- | --- | --- | --- |
| FFT | CNN | 5 | deployment_baseline | 87.87% | 1.2623 |
| FFT | CNN | 10 | deployment_baseline | 87.45% | 1.269 |
| FFT | CNN | 15 | deployment_baseline | 89.03% | 1.2205 |
| FFT | MAML | 5 | deployment_baseline | 99.58% | 1.1955 |
| FFT | MAML | 10 | deployment_baseline | 99.26% | 1.2291 |
| FFT | MAML | 15 | deployment_baseline | 100.0% | 1.3629 |
| FFT | ProtoNet | 5 | deployment_baseline | 100.0% | 1.1701 |
| FFT | ProtoNet | 10 | deployment_baseline | 100.0% | 1.097 |
| FFT | ProtoNet | 15 | deployment_baseline | 100.0% | 1.2365 |
| STFT | CNN | 5 | deployment_baseline | 97.47% | 2.7307 |
| STFT | CNN | 10 | deployment_baseline | 94.62% | 3.0077 |
| STFT | CNN | 15 | deployment_baseline | 97.47% | 2.5726 |
| STFT | MAML | 5 | deployment_baseline | 99.47% | 2.6803 |
| STFT | MAML | 10 | deployment_baseline | 99.89% | 2.3696 |
| STFT | MAML | 15 | deployment_baseline | 100.0% | 2.6757 |
| STFT | ProtoNet | 5 | deployment_baseline | 100.0% | 2.686 |
| STFT | ProtoNet | 10 | deployment_baseline | 100.0% | 2.5214 |
| STFT | ProtoNet | 15 | deployment_baseline | 100.0% | 2.7864 |
| WT | CNN | 5 | deployment_baseline | 94.83% | 2.6458 |
| WT | CNN | 10 | deployment_baseline | 94.09% | 2.479 |
| WT | CNN | 15 | deployment_baseline | 94.83% | 2.4696 |
| WT | MAML | 5 | deployment_baseline | 99.79% | 2.7593 |
| WT | MAML | 10 | deployment_baseline | 99.26% | 2.6715 |
| WT | MAML | 15 | deployment_baseline | 99.89% | 2.5744 |
| WT | ProtoNet | 5 | deployment_baseline | 100.0% | 2.5578 |
| WT | ProtoNet | 10 | deployment_baseline | 100.0% | 2.6533 |
| WT | ProtoNet | 15 | deployment_baseline | 100.0% | 2.7214 |

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
