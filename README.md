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

## 环境依赖

推荐 Python 版本：

- `Python 3.9`

推荐先创建独立环境：

```bash
conda create -n fault_env python=3.9 -y
conda activate fault_env
```

### 安装方式

#### 方式一：CPU 环境

```bash
pip install -r requirements.txt
```

#### 方式二：CUDA 环境

如果你的机器有 NVIDIA GPU，建议先按 PyTorch 官方方式安装与你 CUDA 版本匹配的 `torch` / `torchvision`，再安装其余依赖。

例如 `CUDA 11.8`：

```bash
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

如果你已经提前装好了 `torch` 和 `torchvision`，也可以直接：

```bash
pip install -r requirements.txt
```

### 主要依赖说明

- `torch` / `torchvision`：训练与推理主框架
- `learn2learn`：MAML / ProtoNet 元学习任务构造
- `onnx`：模型导出
- `onnxruntime`：ONNX 推理与量化
- `matplotlib`：训练曲线绘制
- `h5py`、`scipy`、`PyWavelets`：故障信号读取与预处理

### 常见依赖问题

#### 1. `ModuleNotFoundError: No module named 'learn2learn'`

直接执行：

```bash
pip install learn2learn==0.2.0
```

#### 2. `ModuleNotFoundError: No module named 'matplotlib'`

直接执行：

```bash
pip install matplotlib==3.4.3
```

#### 3. `onnxruntime quantization is unavailable`

说明 `onnxruntime` 没装好，执行：

```bash
pip install onnx==1.16.1 onnxruntime==1.18.1
```

#### 4. `pkg_resources is deprecated`

这个通常来自 `setuptools` 版本过新，当前项目已经在 `requirements.txt` 中限制：

```text
setuptools<81
```

#### 5. `Gym has been unmaintained...`

这条警告通常来自 `learn2learn` 的间接依赖，目前不会阻止训练。只要程序能正常开始训练，可以先忽略。

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
- `CWRU` 默认行为：从 10 类标签池中采样 `5-way` 任务

如果要固定 5 类实验，可以显式传：

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

## Contributors

- StarryCode-Lang

## License

MIT
