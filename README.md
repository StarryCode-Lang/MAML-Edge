# MAML-Edge

工业设备少样本故障诊断项目，按四层组织：

- `data_layer`：数据加载与预处理
- `model_layer`：`MAML`、`ProtoNet`、`CNN`
- `deploy_layer`：结构化剪枝、恢复微调、ONNX 导出、INT8 PTQ
- `test_layer`：结果检查

根目录统一入口：

```bash
python train.py --mode {train|deploy} --algorithm {maml|protonet|cnn} ...
```

部署层内部也可单独运行：

```bash
python deploy_layer/deploy.py --algorithm {maml|protonet|cnn} ...
```

## 环境

推荐：

- `Python 3.9`
- GPU 环境先按 PyTorch 官方方式安装 `torch`

安装：

```bash
conda create -n fault_env python=3.9 -y
conda activate fault_env
pip install -r requirements.txt
```

## 数据目录

默认数据目录：`./data`

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

## 默认设置

- `ways = 5`
- `shots = 5`
- `query_shots = shots`
- `CWRU` 默认标签池：`0,1,2,3,4,5,6,7,8,9`
- `HST` 默认标签池：`0,2,3,5,6`
- `plot_step`、`checkpoint_step` 默认按总步数的 `1/5` 自动设置
- `runtime_backend` 默认使用 `onnxruntime`

固定 5 类标签时可显式传入：

```bash
--fault_labels 0,1,2,3,4
```

## 部署后端

- `onnxruntime`：默认主方案
- `tensorrt`：NVIDIA 平台可选
- `openvino`：Intel 平台可选

## 从头训练并完成压缩导出

现在代码支持从训练开始一路跑到：

- 结构化剪枝
- 恢复微调
- ONNX 导出
- INT8 PTQ

训练命令中加入：

```bash
--enable_compression True --prune_ratio 0.4
```

### FFT

#### MAML

```bash
python train.py --mode train --algorithm maml \
  --runtime_backend onnxruntime \
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

#### ProtoNet

```bash
python train.py --mode train --algorithm protonet \
  --runtime_backend onnxruntime \
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

#### CNN

```bash
python train.py --mode train --algorithm cnn \
  --runtime_backend onnxruntime \
  --dataset CWRU \
  --preprocess FFT \
  --ways 5 \
  --shots 5 \
  --query_shots 5 \
  --train_domains 0,1,2 \
  --test_domain 3 \
  --epochs 50 \
  --enable_compression True \
  --prune_ratio 0.4
```

### STFT

#### MAML

```bash
python train.py --mode train --algorithm maml \
  --runtime_backend onnxruntime \
  --dataset CWRU \
  --preprocess STFT \
  --ways 5 \
  --shots 5 \
  --train_domains 0,1,2 \
  --test_domain 3 \
  --iters 100 \
  --enable_compression True \
  --prune_ratio 0.4
```

#### ProtoNet

```bash
python train.py --mode train --algorithm protonet \
  --runtime_backend onnxruntime \
  --dataset CWRU \
  --preprocess STFT \
  --ways 5 \
  --shots 5 \
  --train_domains 0,1,2 \
  --test_domain 3 \
  --iters 100 \
  --enable_compression True \
  --prune_ratio 0.4
```

#### CNN

```bash
python train.py --mode train --algorithm cnn \
  --runtime_backend onnxruntime \
  --dataset CWRU \
  --preprocess STFT \
  --ways 5 \
  --shots 5 \
  --query_shots 5 \
  --train_domains 0,1,2 \
  --test_domain 3 \
  --epochs 50 \
  --enable_compression True \
  --prune_ratio 0.4
```

## 从已有 checkpoint 直接做压缩导出

如果 `checkpoints/` 下已经有 `*_best.pt`，可以跳过训练，直接部署：

### MAML

```bash
python train.py --mode deploy --algorithm maml \
  --runtime_backend onnxruntime \
  --dataset CWRU \
  --preprocess STFT \
  --ways 5 \
  --shots 5 \
  --train_domains 0,1,2 \
  --test_domain 3 \
  --prune_ratio 0.4 \
  --compression_finetune_iters 40
```

### ProtoNet

```bash
python train.py --mode deploy --algorithm protonet \
  --runtime_backend onnxruntime \
  --dataset CWRU \
  --preprocess STFT \
  --ways 5 \
  --shots 5 \
  --query_shots 5 \
  --train_domains 0,1,2 \
  --test_domain 3 \
  --prune_ratio 0.4 \
  --compression_finetune_iters 40
```

如果 checkpoint 或 history 不在默认位置，可额外指定：

```bash
--best_checkpoint_path checkpoints/xxx_best.pt
--history_path checkpoints/xxx_history.json
```

## 输出目录

输出位于：

```text
deploy_artifacts/<experiment_title>/
```

主要文件：

- `compression_summary.json`
- `*_float.onnx`
- `*_int8.onnx`
- `ProtoNet` 额外输出 `*_prototypes.npz`

## 测试脚本

```bash
python test_layer/benchmark.py \
  --summary_path deploy_artifacts/<experiment_title>/compression_summary.json
```

## 参考资料

- [fyancy/MetaFD](https://github.com/fyancy/MetaFD)
- [Yifei20/Few-shot-Fault-Diagnosis-MAML](https://github.com/Yifei20/Few-shot-Fault-Diagnosis-MAML)
- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)
- [Meta-learning as a promising approach for few-shot cross-domain fault diagnosis: Algorithms, applications, and prospects](https://doi.org/10.1016/j.knosys.2021.107646)

## Contributors

- StarryCode-Lang

## License

MIT
