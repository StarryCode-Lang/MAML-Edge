# MAML-Edge

面向工业设备少样本故障诊断的元学习项目，按四层结构组织：

- `data_layer`：CWRU / HST 数据加载与预处理
- `model_layer`：`MAML`、`ProtoNet`、`CNN`
- `deploy_layer`：结构化剪枝、恢复微调、ONNX 导出、INT8 PTQ
- `test_layer`：部署结果指标检查

统一训练入口：

```bash
python train.py --algorithm {maml|protonet|cnn} ...
```

## 项目结构

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

## 环境安装

```bash
conda create -n maml-edge python=3.9 -y
conda activate maml-edge
pip install -r requirements.txt
```

## 数据目录

默认数据根目录：`./data`

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

## 默认实验设置

- `ways = 5`
- `shots = 5`
- `query_shots = shots`
- 默认标签池：
  - `CWRU: 0,1,2,3,4,5,6,7,8,9`
  - `HST: 0,2,3,5,6`
- `CWRU` 默认行为是：从 10 类标签池中采样 `5-way` 任务

如果你要固定 5 类实验，可以显式传：

```bash
python train.py --algorithm maml --fault_labels 0,1,2,3,4 ...
```

## 训练命令

### FFT

适合轻量化 1D 模型，通常训练轮次更长。

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

适合时频表征更强的设定，通常迭代数可以更低。

#### MAML

```bash
python train.py --algorithm maml \
  --dataset CWRU \
  --preprocess STFT \
  --ways 5 \
  --shots 5 \
  --train_domains 0,1,2 \
  --test_domain 3 \
  --iters 100
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
  --iters 100
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

## 压缩与导出

压缩流程对应最终方案：

```text
结构化剪枝（约 40%，通道级）
-> 恢复微调
-> INT8 PTQ
-> ONNX Runtime 部署
```

示例：

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

输出目录：

```text
deploy_artifacts/<experiment_title>/
```

主要文件：

- 浮点 ONNX 模型
- INT8 ONNX 模型
- ProtoNet 原型文件
- `compression_summary.json`

## 测试脚手架

```bash
python test_layer/benchmark.py \
  --summary_path deploy_artifacts/<experiment_title>/compression_summary.json
```

## 参考资料
- [fyancy/MetaFD](https://github.com/fyancy/MetaFD)
- [Yifei20/Few-shot-Fault-Diagnosis-MAML](https://github.com/Yifei20/Few-shot-Fault-Diagnosis-MAML)
- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)
- [Meta-learning as a promising approach for few-shot cross-domain fault diagnosis: Algorithms, applications, and prospects](https://doi.org/10.1016/j.knosys.2021.107646)
