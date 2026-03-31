# MAML-Edge: Few-Shot Fault Diagnosis with Meta Learning

基于 MAML 和 ProtoNet 的少样本故障诊断框架，适用于边缘计算场景。

## 特性

- **双模型对比**: MAML/FOMAML (主模型) 与 Prototypical Network (对比模型)
- **多数据集支持**: CWRU 轴承数据集、HST 高速列车数据集
- **灵活预处理**: 支持 FFT、STFT、小波变换 (WT)
- **跨域学习**: 多工况源域到目标域的迁移学习
- **确定性复现**: 固定种子和评估池确保实验可重复

## 工作流程

```
数据加载 → 滑动窗口(1024) → 预处理(FFT/STFT/WT) → N-way K-shot 任务构造 → 元学习训练
```

## 快速开始

### 环境安装

```bash
conda create -n fault_env python=3.9 -y
conda activate fault_env
pip install -r requirements.txt
```

### 数据准备

下载 CWRU 数据集并按以下结构组织：

```text
data/
  CWRU_12k/
    Drive_end_0/
    Drive_end_1/
    Drive_end_2/
    Drive_end_3/
```

### 训练 MAML 模型

```bash
python train_maml.py \
  --ways 5 \
  --shots 5 \
  --iters 1500 \
  --first_order True \
  --preprocess FFT \
  --train_domains 0,1,2 \
  --test_domain 3 \
  --meta_batch_size 10 \
  --meta_test_batch_size 3
```

### 训练 ProtoNet 模型

```bash
python train_protonet.py \
  --ways 5 \
  --shots 5 \
  --iters 1500 \
  --preprocess FFT \
  --train_domains 0,1,2 \
  --test_domain 3 \
  --meta_batch_size 10 \
  --meta_test_batch_size 3
```

## 项目结构

| 文件 | 说明 |
|------|------|
| `train_maml.py` | MAML 训练入口 |
| `maml.py` | MAML 核心训练逻辑 |
| `train_protonet.py` | ProtoNet 训练入口 |
| `protonet.py` | ProtoNet 核心训练逻辑 |
| `models.py` | CNN1D/CNN2D 模型定义 |
| `fault_datasets.py` | CWRU/HST 数据集封装 |
| `preprocess_cwru.py` | CWRU 数据加载与预处理 |
| `preprocess_hst.py` | HST 数据加载与预处理 |
| `utils.py` | 工具函数(日志、适应、评估等) |
| `requirements.txt` | Python 依赖 |

## 核心参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--ways` | 每任务类别数 | 5 |
| `--shots` | 支持样本数 | 5 |
| `--query_shots` | 查询样本数 | 与 shots 相同 |
| `--iters` | 元学习迭代次数 | 1000 |
| `--meta_lr` | 外循环学习率 | 0.001 |
| `--fast_lr` | 内循环学习率 | 0.1 |
| `--adapt_steps` | 内循环更新步数 | 5 |
| `--first_order` | 是否使用一阶近似 | True |
| `--preprocess` | 预处理方法 (FFT/STFT/WT) | FFT |
| `--train_domains` | 源域列表 | 0,1,2 |
| `--test_domain` | 目标域 | 3 |
| `--eval_support_ratio` | 目标域评估支持池比例 | 0.5 |

## 模型架构

- **CNN1D** (FFT 预处理): 3层卷积 + BatchNorm + ReLU + MaxPool
- **CNN2D** (STFT/WT 预处理): 4层卷积块 + 自适应池化
- **ProtoNet Encoder**: 对应 CNN 的编码器部分

## 实验设置

标准配置建议：

```bash
# 不同 shot 设置
for shots in 5 10 15; do
  python train_maml.py --ways 5 --shots $shots --iters 1500 \
    --preprocess FFT --train_domains 0,1,2 --test_domain 3 \
    --seed $RANDOM
done

# 重复实验 (>= 5次) 统计稳定性
for seed in 42 123 456 789 1024; do
  python train_maml.py --seed $seed ...
done
```

## 输出说明

训练过程生成以下文件：

- `logs/`: 训练日志
- `images/`: 准确率和损失曲线
- `checkpoints/`: 模型检查点 (.pt 文件)

## 支持的数据集

### CWRU 轴承数据集
- 4种工况 (0/1/2/3 HP)
- 10类故障类型 (正常 + 9种故障)
- 12kHz 采样率

### HST 高速列车数据集
- 3种速度 (20/160/280 km/h)
- 5类故障 (partial mode)
- 支持多传感器通道

## 注意事项

- 目标域评估使用固定的支持池/查询池划分 (`eval_support_ratio`)
- `meta_test_batch_size` 可选，未设置时保持默认行为
- 建议使用 `--first_order True` 加速训练 (FOMAML)
- 不同数据集的输出类别数不同 (CWRU=10, HST=5)

## 参考文献

- [Finn et al. Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)
- [Snell et al. Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175)
- [Few-shot Fault Diagnosis with MAML](https://github.com/Yifei20/Few-shot-Fault-Diagnosis-MAML)
- [MetaFD](https://github.com/fyancy/MetaFD)

## 许可证

本项目采用 MIT 许可证。
