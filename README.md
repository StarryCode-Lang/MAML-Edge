# MAML-Edge

Few-shot industrial fault diagnosis project built around four layers:

- `data_layer`: CWRU and HST data loading, preprocessing, and fault subset control
- `model_layer`: MAML, ProtoNet, and traditional CNN baseline
- `deploy_layer`: structured pruning, recovery fine-tuning, ONNX export, INT8 PTQ, optional QAT
- `test_layer`: benchmark summary checking

This branch follows the final project scheme:

```text
Structured pruning (~40%, channel level)
-> recovery fine-tuning
-> INT8 post-training quantization
-> ONNX Runtime deployment
```

## Project Layout

```text
MAML-Edge/
|-- data_layer/
|   |-- fault_datasets.py
|   |-- preprocess_cwru.py
|   `-- preprocess_hst.py
|-- model_layer/
|   |-- experiment.py
|   |-- models.py
|   |-- utils.py
|   |-- maml.py
|   |-- protonet.py
|   |-- cnn_baseline.py
|   |-- train_maml.py
|   |-- train_protonet.py
|   `-- train_cnn.py
|-- deploy_layer/
|   `-- compression.py
|-- test_layer/
|   `-- benchmark.py
|-- train_maml.py
|-- train_protonet.py
|-- train_cnn.py
|-- requirements.txt
`-- README.md
```

Only the three train entry scripts are kept at repo root. Real implementation stays inside the four layers.

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
- target-domain evaluation uses fixed support/query pools

You can override the fault subset with:

```bash
python train_maml.py --fault_labels 0,1,2,3,4
```

## Controlled Variables

The current branch standardizes the comparison setup for `CNN / MAML / ProtoNet` as much as possible without changing each algorithm's nature.

Shared across all three:

- same dataset and same selected fault subset
- same `train_domains` and `test_domain`
- same `ways / shots / query_shots`
- same target-domain fixed support/query pool evaluation rule
- same backbone width configuration
- same best-checkpoint selection rule

Shared backbone defaults:

- `FFT`: `32,64,64` with `AdaptiveAvgPool1d(64)`
- `STFT/WT`: `64,64,64,64`

Algorithm-specific parts intentionally remain different:

- `MAML`: `fast_lr`, `adapt_steps`, `first_order`
- `ProtoNet`: prototype-based distance classification
- `CNN`: supervised source training, but target evaluation is now also few-shot episodic for fairer comparison

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

This lets you change backbone capacity once and keep the comparison aligned.

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

Artifacts are written to:

```text
deploy_artifacts/<experiment_title>/
```

Typical outputs:

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
