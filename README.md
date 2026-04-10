# MAML-Edge

工业物联网下基于元学习的边缘设备少样本故障诊断系统。

当前 `codex/thesis-upgrade` 分支已经收敛为一个**可复现、可对比、可写论文**的最终实验版本，目标是稳定输出：

- 算法对比结果
- 少样本能力结果
- 压缩部署结果
- 系统在线性能结果

## 项目结构

- `data_layer`：CWRU / HST 数据读取、FFT / STFT / WT 预处理、few-shot 数据构造
- `model_layer`：`MAML`、`ProtoNet`、`CNN` 的训练与评估
- `deploy_layer`：结构化剪枝、恢复微调、ONNX 导出、INT8 PTQ、推理后端选择
- `test_layer`：benchmark、实验矩阵、结果聚合、论文表格导出
- `edge_layer`：边缘侧信号模拟、轻量预处理、MQTT 上传
- `system_layer`：FastAPI、MQTT、WebSocket、历史/报警存储与 Vue + ECharts 控制台

统一训练入口：

```bash
python train.py --mode {train|deploy} --algorithm {maml|protonet|cnn} ...
```

## 环境

完整训练环境：

```bash
conda create -n fault_env python=3.9 -y
conda activate fault_env
pip install -r requirements.txt
```

后两层最小联调环境：

```bash
conda create -n edge_system python=3.9 -y
conda activate edge_system
pip install -r requirements.edge-system.txt
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

## 当前最终实验配置

论文主实验固定为：

- `Dataset = CWRU`
- `Preprocess = STFT`
- `Task = 5-way 5-shot`
- `Train domains = 0,1,2`
- `Test domain = 3`
- `Runtime backend = onnxruntime`
- `Structured pruning = 0.4`
- `INT8 = PTQ`

不再展开：

- 不再扫描不同剪枝率
- 不再把 `TensorRT / OpenVINO` 作为主实验后端
- 不再增加新的系统功能

## 当前主实验设计

### 实验 1：模型对比

比较：

- `CNN`
- `MAML`
- `ProtoNet`

输出：

- 准确率

### 实验 2：少样本能力

固定主模型：

- `MAML`

比较：

- `5-shot`
- `10-shot`
- `15-shot`

输出：

- 准确率

### 实验 3：压缩影响

固定主模型：

- `MAML`

比较：

- `Original`
- `Pruned`
- `Pruned + INT8`

输出：

- 准确率
- 平均推理时延
- 参数量
- 模型大小

### 实验 4：系统性能

系统链路固定为：

```text
Edge Simulator -> MQTT -> FastAPI -> ONNX -> WebSocket -> Frontend
```

输出：

- `preprocess_latency_ms`
- `inference_latency_ms`
- `end_to_end_latency_ms`

### 实验 5：最小在线适配

当前系统支持最小可用的运行时原型更新：

- 接口：`POST /adapt`
- 适用对象：`ProtoNet` 类型的 prototype deployment

它用于展示“少量 support sample 的运行时原型更新”，不是完整在线再训练系统。

## 默认训练与部署命令

### CNN

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
  --epochs 60 \
  --enable_compression true \
  --prune_ratio 0.4
```

### MAML

```bash
python train.py --mode train --algorithm maml \
  --runtime_backend onnxruntime \
  --dataset CWRU \
  --preprocess STFT \
  --ways 5 \
  --shots 5 \
  --train_domains 0,1,2 \
  --test_domain 3 \
  --iters 200 \
  --enable_compression true \
  --prune_ratio 0.4
```

### ProtoNet

```bash
python train.py --mode train --algorithm protonet \
  --runtime_backend onnxruntime \
  --dataset CWRU \
  --preprocess STFT \
  --ways 5 \
  --shots 5 \
  --query_shots 5 \
  --train_domains 0,1,2 \
  --test_domain 3 \
  --iters 200 \
  --enable_compression true \
  --prune_ratio 0.4
```

## deploy_artifacts 输出

每次训练并开启压缩后，会输出到：

```text
deploy_artifacts/<experiment_title>/
```

主要文件：

- `compression_summary.json`
- `*_baseline_float.onnx`
- `*_float.onnx`
- `*_int8.onnx`
- `ProtoNet` 额外输出 `*_prototypes.npz`

其中 `compression_summary.json` 现在会统一记录：

- 训练实验描述
- 部署后端与部署类型
- 原始模型、剪枝模型、INT8 模型的精度与时延
- 参数量与压缩比例
- 各部署产物的文件大小

## test_layer 怎么用

`test_layer` 现在不是通用测试杂项，而是**运行结果分析与论文出表层**。

核心文件：

- [benchmark.py](/D:/Desktop/MAML-Edge/test_layer/benchmark.py)
- [result_aggregator.py](/D:/Desktop/MAML-Edge/test_layer/result_aggregator.py)
- [run_controlled_overnight.sh](/D:/Desktop/MAML-Edge/test_layer/run_controlled_overnight.sh)
- [thesis_config.py](/D:/Desktop/MAML-Edge/test_layer/thesis_config.py)
- [thesis_tables.py](/D:/Desktop/MAML-Edge/test_layer/thesis_tables.py)

约束边界：

- `run_controlled_overnight.sh` 负责执行你原本会在终端里手动输入的训练与部署命令
- `test_layer` 中的 Python 文件只负责读取运行后的 `logs/`、`deploy_artifacts/`、`checkpoints/` 并做分析导出

### 1. 一条命令跑完受控过夜实验

```bash
bash test_layer/run_controlled_overnight.sh restart
```

这个入口会顺序执行：

- `CNN / MAML / ProtoNet`
- `FFT / STFT / WT`
- `5-shot / 10-shot / 15-shot`
- 自动训练、剪枝恢复、ONNX 导出、INT8 PTQ、benchmark 聚合
- 自动导出 `logs/thesis_tables/controlled/` 下的表格文件

完整矩阵共 `27` 组：

- `FFT + CNN/MAML/ProtoNet + 5/10/15-shot`
- `STFT + CNN/MAML/ProtoNet + 5/10/15-shot`
- `WT + CNN/MAML/ProtoNet + 5/10/15-shot`

受控统一调度：

- `FFT`: `MAML / ProtoNet iters = 400`
- `STFT`: `MAML / ProtoNet iters = 80`
- `WT`: `MAML / ProtoNet iters = 80`
- `CNN epochs`: `FFT = 40`, `STFT = 30`, `WT = 30`
- `test_task_num = 50`
- `compression_finetune_iters = 80`

日志按预处理和算法分类保存：

```text
logs/overnight_runs/controlled/latest/logs/<preprocess>/<algorithm>/
```

如需先查看脚本里会顺序执行的全部终端命令：

`run_controlled_overnight.sh` 现只支持 `clean | run | restart`。

论文增强实验入口：

```bash
bash test_layer/run_seed_extension.sh restart
bash test_layer/run_domain_extension.sh restart
bash test_layer/run_compression_ablation.sh restart
bash test_layer/run_paper_suite.sh restart
```

### 2. 导出单实验 benchmark

```bash
python test_layer/benchmark.py \
  --summary_path deploy_artifacts/<experiment_title>/compression_summary.json
```

### 3. 导出单实验论文行

```bash
python test_layer/benchmark.py \
  --summary_path deploy_artifacts/<experiment_title>/compression_summary.json \
  --rows_only \
  --output_format csv \
  --output_path logs/<experiment_title>_benchmark_row.csv
```

### 4. 聚合全部部署结果

```bash
python test_layer/result_aggregator.py \
  --summary_glob "deploy_artifacts/*/compression_summary.json" \
  --output_format csv \
  --output_path logs/thesis_benchmark_rows.csv
```

### 5. 直接导出论文四张表

```bash
python test_layer/thesis_tables.py \
  --summary_glob "deploy_artifacts/*/compression_summary.json" \
  --output_dir logs/thesis_tables
```

它会输出：

- `table1_model_performance.csv`
- `table2_few_shot.csv`
- `table3_compression.csv`
- `thesis_tables.json`
- `thesis_tables.md`

其中 `thesis_tables.md` 是论文草稿和答辩整理最方便看的版本。

### 6. 导出论文增强版统计表

```bash
python test_layer/paper_tables.py \
  --output_dir logs/thesis_tables/paper_balanced
```

### 6. 导出系统性能表

先启动系统并让它跑过一轮 MQTT 联调，再执行：

```bash
python test_layer/thesis_tables.py \
  --summary_glob "deploy_artifacts/*/compression_summary.json" \
  --system_stats_url http://127.0.0.1:8000/system/stats \
  --system_channel mqtt \
  --output_dir logs/thesis_tables
```

这会额外生成：

- `table4_system_performance.csv`

## 各类指标来源

### 训练指标

来源：

- `checkpoints/*_history.json`

用于：

- 算法精度对比
- 学习曲线
- 最佳 checkpoint 选择

### 部署指标

来源：

- `deploy_artifacts/*/compression_summary.json`

用于：

- 原始 / 剪枝 / INT8 对比
- 压缩前后参数量对比
- 模型大小与平均推理时延分析

### 系统级在线指标

来源：

- `GET /system/stats`
- `GET /history`
- `GET /alerts`
- `WS /ws/realtime`

关键字段：

- `preprocess_latency_ms`
- `inference_latency_ms`
- `end_to_end_latency_ms`

语义：

- `inference_latency_ms`：部署层纯推理时延
- `preprocess_latency_ms`：系统层在线预处理时延
- `end_to_end_latency_ms`：系统层在线端到端时延
- `latency_ms`：兼容字段，等于 `end_to_end_latency_ms`

## system_layer 联调

推荐显式指定模型：

```bash
MAML_EDGE_MODEL_SUMMARY_PATH=deploy_artifacts/<experiment_title>/compression_summary.json
```

启动后端：

```bash
python -m uvicorn system_layer.backend.main:app --host 0.0.0.0 --port 8000
```

启动 MQTT：

```bash
mosquitto -p 1883
```

启动边缘模拟器：

```bash
python -m edge_layer.simulator.publish_signal \
  --source synthetic \
  --device_id esp32-sim-01 \
  --host 127.0.0.1 \
  --port 1883 \
  --topic maml-edge/devices/esp32-sim-01/signal \
  --count 5 \
  --interval 1.0
```

浏览器控制台：

```text
http://127.0.0.1:8000/
```

主要接口：

- `GET /health`
- `GET /model/info`
- `GET /benchmark/current`
- `GET /system/stats`
- `GET /history`
- `GET /alerts`
- `POST /predict`
- `POST /adapt`
- `POST /simulate/publish`
- `WS /ws/realtime`

## 论文表述边界

### 已完成

- 六层一体化结构
- `CNN / MAML / ProtoNet` 训练与评估
- 剪枝 + ONNX + INT8 部署
- FastAPI + MQTT + WebSocket 在线系统
- Vue + ECharts 浏览器控制台
- 最小可用的 ProtoNet runtime prototype update
- 固定 thesis 实验矩阵与四张论文表导出

### 已支持或预留，但不作为主实验

- `openvino`
- `tensorrt`

### 不应写成“完整实现”

- 复杂 `STM32 / ESP32` 实机工程
- 完整在线再训练系统

## 参考资料

- [fyancy/MetaFD](https://github.com/fyancy/MetaFD)
- [Yifei20/Few-shot-Fault-Diagnosis-MAML](https://github.com/Yifei20/Few-shot-Fault-Diagnosis-MAML)
- [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)
- [Meta-learning as a promising approach for few-shot cross-domain fault diagnosis: Algorithms, applications, and prospects](https://doi.org/10.1016/j.knosys.2021.107646)
- [Prototypical Networks for Few-shot Learning](https://arxiv.org/pdf/1703.05175)
- [Matching Networks for One Shot Learning](https://arxiv.org/pdf/1606.04080)
- [CWRU official Database](https://engineering.case.edu/bearingdatacenter/download-data-file)

## License

MIT
