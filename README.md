# MAML-Edge

Few-shot industrial fault diagnosis project organized into four layers:

- `data_layer`: CWRU and HST data loading, preprocessing, and fault subset control
- `model_layer`: MAML, ProtoNet, and CNN baseline
- `deploy_layer`: structured pruning, recovery fine-tuning, ONNX export, INT8 PTQ, optional QAT
- `test_layer`: benchmark summary checking

The project follows the final scheme:

```text
Structured pruning (~40%, channel level)
-> recovery fine-tuning
-> INT8 post-training quantization
-> ONNX Runtime deployment
-> MQTT / web diagnostic system integration
```

## Project Layout

```text
MAML-Edge/
|-- data_layer/
|   |-- fault_datasets.py
|   |-- preprocess_cwru.py
|   |-- preprocess_hst.py
|   `-- __init__.py
|-- model_layer/
|   |-- experiment.py
|   |-- models.py
|   |-- utils.py
|   |-- maml.py
|   |-- protonet.py
|   |-- cnn_baseline.py
|   |-- train_maml.py
|   |-- train_protonet.py
|   |-- train_cnn.py
|   `-- __init__.py
|-- deploy_layer/
|   |-- compression.py
|   `-- __init__.py
|-- test_layer/
|   |-- benchmark.py
|   `-- __init__.py
|-- train_maml.py
|-- train_protonet.py
|-- train_cnn.py
|-- requirements.txt
|-- LICENSE
`-- README.md
```

Only the three training entry scripts are kept at the repository root. Main implementation stays inside the four layers.

## Environment

```bash
conda create -n maml-edge python=3.9 -y
conda activate maml-edge
pip install -r requirements.txt
```

## Data

Default data root is `./data`.

### CWRU

```text
data/
`-- CWRU_12k/
    |-- Drive_end_0/
    |-- Drive_end_1/
    |-- Drive_end_2/
    `-- Drive_end_3/
```

### HST

```text
data/
`-- HST/
    |-- 0/
    |-- 1/
    `-- 2/
```

## Default Experiment Setup

- `ways = 5`
- `shots = 5`
- `query_shots = shots`
- default fault subsets:
  - `CWRU: 0,1,2,3,4`
  - `HST: 0,2,3,5,6`
- target-domain evaluation uses a fixed support/query pool

You can override the fault subset with:

```bash
python train_maml.py --fault_labels 0,1,2,3,4
```

## Controlled Variables

This branch aligns `CNN / MAML / ProtoNet` as much as possible without changing their algorithmic nature.

Shared across all three:

- same dataset and selected fault subset
- same `train_domains` and `test_domain`
- same `ways / shots / query_shots`
- same target-domain fixed-pool evaluation rule
- same backbone width configuration
- same best-checkpoint selection rule

Shared backbone defaults:

- `FFT`: `32,64,64` with `AdaptiveAvgPool1d(64)`
- `STFT/WT`: `64,64,64,64`

Algorithm-specific parts intentionally remain different:

- `MAML`: `fast_lr`, `adapt_steps`, `first_order`
- `ProtoNet`: prototype-based distance classification
- `CNN`: supervised source-domain training with target-domain few-shot episodic evaluation

## Training

### MAML

```bash
python train_maml.py \
  --dataset CWRU \
  --preprocess FFT \
  --ways 5 \
  --shots 5 \
  --train_domains 0,1,2 \
  --test_domain 3 \
  --iters 1500
```

### ProtoNet

```bash
python train_protonet.py \
  --dataset CWRU \
  --preprocess FFT \
  --ways 5 \
  --shots 5 \
  --train_domains 0,1,2 \
  --test_domain 3 \
  --iters 1500
```

### CNN Baseline

```bash
python train_cnn.py \
  --dataset CWRU \
  --preprocess FFT \
  --ways 5 \
  --shots 5 \
  --query_shots 5 \
  --train_domains 0,1,2 \
  --test_domain 3 \
  --epochs 50
```

## Shared Backbone Parameters

All three algorithms support the same backbone control arguments:

```bash
--fft_channels 32,64,64
--image_channels 64,64,64,64
--fft_pooled_length 64
```

## Compression and Export

Enable the compression pipeline from the training entry:

```bash
python train_maml.py \
  --dataset CWRU \
  --preprocess FFT \
  --ways 5 \
  --shots 5 \
  --train_domains 0,1,2 \
  --test_domain 3 \
  --iters 1500 \
  --enable_compression True \
  --prune_ratio 0.4
```

Typical outputs are written to:

```text
deploy_artifacts/<experiment_title>/
```

Including:

- float ONNX model
- INT8 ONNX model
- ProtoNet prototype file
- `compression_summary.json`

## Best Model Selection

Training keeps the best model using:

1. `meta_test_acc`
2. `meta_test_loss`
3. `meta_train_acc`

By default only the best checkpoint is kept. To keep all intermediate checkpoints:

```bash
--keep_all_checkpoints True
```

## Benchmark Helper

```bash
python test_layer/benchmark.py \
  --summary_path deploy_artifacts/<experiment_title>/compression_summary.json
```

Default checks:

- accuracy >= `0.95`
- average latency <= `100 ms`

## Dependencies

Core packages are listed in `requirements.txt`, including:

- `torch`
- `learn2learn`
- `matplotlib`
- `onnx`
- `onnxruntime`

## Contributors

- StarryCode-Lang

## License

MIT
