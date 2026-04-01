# MAML-Edge

Few-shot industrial fault diagnosis project with four layers:

- `data_layer`: CWRU / HST loading and preprocessing
- `model_layer`: `MAML`, `ProtoNet`, `CNN`
- `deploy_layer`: pruning, recovery fine-tuning, ONNX export, INT8 PTQ
- `test_layer`: benchmark summary checking

Unified root entry:

```bash
python train.py --algorithm {maml|protonet|cnn} ...
```

## Structure

```text
MAML-Edge/
|-- data_layer/
|-- model_layer/
|-- deploy_layer/
|-- test_layer/
|-- train.py
|-- requirements.txt
|-- LICENSE
`-- README.md
```

## Environment

```bash
conda create -n maml-edge python=3.9 -y
conda activate maml-edge
pip install -r requirements.txt
```

## Data

Default data root: `./data`

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

## Defaults

- `ways = 5`
- `shots = 5`
- `query_shots = shots`
- default label pool:
  - `CWRU: 0,1,2,3,4,5,6,7,8,9`
  - `HST: 0,2,3,5,6`
- `CWRU` default behavior is: sample `5-way` episodes from the full `10-label` pool

If you need a fixed 5-class experiment:

```bash
python train.py --algorithm maml --fault_labels 0,1,2,3,4 ...
```

## Training

### FFT

Recommended when you want a lighter 1D model and longer training.

#### MAML

```bash
python train.py --algorithm maml \
  --dataset CWRU \
  --preprocess FFT \
  --ways 5 \
  --shots 5 \
  --train_domains 0,1,2 \
  --test_domain 3 \
  --iters 1500
```

#### ProtoNet

```bash
python train.py --algorithm protonet \
  --dataset CWRU \
  --preprocess FFT \
  --ways 5 \
  --shots 5 \
  --train_domains 0,1,2 \
  --test_domain 3 \
  --iters 1500
```

#### CNN

```bash
python train.py --algorithm cnn \
  --dataset CWRU \
  --preprocess FFT \
  --ways 5 \
  --shots 5 \
  --query_shots 5 \
  --train_domains 0,1,2 \
  --test_domain 3 \
  --epochs 50
```

### STFT

Recommended when you want stronger time-frequency representation and usually fewer iterations.

#### MAML

```bash
python train.py --algorithm maml \
  --dataset CWRU \
  --preprocess STFT \
  --ways 5 \
  --shots 5 \
  --train_domains 0,1,2 \
  --test_domain 3 \
  --iters 200
```

#### ProtoNet

```bash
python train.py --algorithm protonet \
  --dataset CWRU \
  --preprocess STFT \
  --ways 5 \
  --shots 5 \
  --train_domains 0,1,2 \
  --test_domain 3 \
  --iters 200
```

#### CNN

```bash
python train.py --algorithm cnn \
  --dataset CWRU \
  --preprocess STFT \
  --ways 5 \
  --shots 5 \
  --query_shots 5 \
  --train_domains 0,1,2 \
  --test_domain 3 \
  --epochs 50
```

## Auto Schedule

If `plot_step` and `checkpoint_step` are not passed, they are set automatically to one-fifth of total training steps:

- `MAML / ProtoNet`: `iters // 5`
- `CNN`: `epochs // 5`

Examples:

- `FFT, iters=1500` -> `plot_step=300`, `checkpoint_step=300`
- `STFT, iters=200` -> `plot_step=40`, `checkpoint_step=40`
- `CNN, epochs=50` -> `plot_step=10`, `checkpoint_step=10`

## Backbone

Shared backbone defaults:

- `FFT`: `--fft_channels 32,64,64 --fft_pooled_length 64`
- `STFT/WT`: `--image_channels 64,64,64,64`

## Compression

Compression pipeline matches the final scheme:

```text
Structured pruning (~40%, channel level)
-> recovery fine-tuning
-> INT8 PTQ
-> ONNX Runtime deployment
```

Example:

```bash
python train.py --algorithm maml \
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

Outputs are written to:

```text
deploy_artifacts/<experiment_title>/
```

Main artifacts:

- float ONNX model
- INT8 ONNX model
- ProtoNet prototype file
- `compression_summary.json`

## Benchmark

```bash
python test_layer/benchmark.py \
  --summary_path deploy_artifacts/<experiment_title>/compression_summary.json
```

Default checks:

- accuracy >= `0.95`
- average latency <= `100 ms`

## Dependencies

Core packages in `requirements.txt`:

- `torch`
- `learn2learn`
- `matplotlib`
- `onnx`
- `onnxruntime`

## Contributors

- StarryCode-Lang

## License

MIT
