# MAML-Edge

面向工业故障诊断的小样本元学习项目，当前聚焦 4 个层次：

- 数据层：CWRU 与 HST 数据加载、预处理、5 类故障子集组织
- 模型层：MAML、ProtoNet、传统 CNN 基线
- 部署层：结构化剪枝、恢复微调、ONNX 导出、INT8 PTQ、可选 QAT
- 测试层：部署结果读取与指标阈值检查脚手架

## 项目结构

```text
MAML-Edge/
├─ data_layer/
│  ├─ fault_datasets.py
│  ├─ preprocess_cwru.py
│  └─ preprocess_hst.py
├─ model_layer/
│  ├─ models.py
│  ├─ utils.py
│  ├─ maml.py
│  ├─ protonet.py
│  ├─ cnn_baseline.py
│  ├─ train_maml.py
│  ├─ train_protonet.py
│  └─ train_cnn.py
├─ deploy_layer/
│  └─ compression.py
├─ test_layer/
│  └─ benchmark.py
├─ train_maml.py
├─ train_protonet.py
├─ train_cnn.py
├─ requirements.txt
└─ README.md
```

根目录只保留训练入口脚本，实际实现全部在四层目录中。

## 环境安装

```bash
conda create -n maml-edge python=3.9 -y
conda activate maml-edge
pip install -r requirements.txt
```

## 数据准备

默认数据根目录为 `./data`。

### CWRU

```text
data/
└─ CWRU_12k/
   ├─ Drive_end_0/
   ├─ Drive_end_1/
   ├─ Drive_end_2/
   └─ Drive_end_3/
```

### HST

```text
data/
└─ HST/
   ├─ 0/
   ├─ 1/
   └─ 2/
```

## 默认实验设定

- 默认 `5-way`
- 默认故障子集：
  - CWRU: `0,1,2,3,4`
  - HST: `0,2,3,5,6`
- 支持 `5 / 10 / 15 shot`
- 默认使用目标域固定 support/query 池评估

如需自定义故障类，可通过 `--fault_labels` 指定，例如：

```bash
python train_maml.py --fault_labels 0,1,2,3,4
```

## 训练

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

### 传统 CNN 基线

```bash
python train_cnn.py \
  --dataset CWRU \
  --preprocess FFT \
  --ways 5 \
  --shots 5 \
  --train_domains 0,1,2 \
  --test_domain 3 \
  --epochs 50
```

## 压缩与导出

压缩流程已经接入训练入口，符合当前项目的最终方案：

```text
结构化剪枝（约 40%，通道级）
→ 短周期恢复微调
→ ONNX 导出
→ INT8 PTQ
→ 必要时短周期 QAT
```

开启方式：

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

压缩结果默认输出到：

```text
deploy_artifacts/<experiment_title>/
```

其中包含：

- 最终浮点 ONNX 模型
- INT8 ONNX 模型
- ProtoNet 原型文件
- `compression_summary.json`

## 最佳模型策略

训练过程中不再只保留固定步长的粗放导出，而是会根据以下指标自动记录最佳模型：

- `meta_test_acc` 优先
- `meta_test_loss` 次优先
- `meta_train_acc` 作为补充参考

默认只保留最佳模型；如需保留所有中间 checkpoint，可显式传入：

```bash
--keep_all_checkpoints True
```

## 测试层脚手架

当前没有自动跑完整测试流程，但已提供阈值检查入口：

```bash
python test_layer/benchmark.py --summary_path deploy_artifacts/<experiment_title>/compression_summary.json
```

默认检查：

- 准确率是否达到 `0.95`
- 平均推理时延是否不高于 `100 ms`

## 依赖

核心依赖见 `requirements.txt`：

- `torch`
- `learn2learn`
- `matplotlib`
- `onnx`
- `onnxruntime`

## Contributors

- StarryCode-Lang

## License

MIT
