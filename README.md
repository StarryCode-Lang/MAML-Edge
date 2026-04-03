# MAML-Edge

MAML-Edge 是一个面向本科毕业设计的六层一体化故障诊断项目，目标是把：

- 少样本故障诊断训练
- 轻量化压缩与 ONNX / INT8 部署
- 边缘侧数据接入
- 系统侧实时诊断、报警与可视化

放在同一条完整链路里跑通。

---

## 1. 六层结构

```text
data_layer     -> 数据读取与预处理
model_layer    -> MAML / ProtoNet / CNN 训练
deploy_layer   -> 剪枝、导出、量化、部署推理
test_layer     -> benchmark、实验矩阵、结果聚合
edge_layer     -> 模拟器、MQTT 发布、硬件接口预留
system_layer   -> FastAPI + MQTT + WebSocket + Vue + ECharts 控制台
```

对应目录：

- [data_layer](/D:/Desktop/MAML-Edge/data_layer)
- [model_layer](/D:/Desktop/MAML-Edge/model_layer)
- [deploy_layer](/D:/Desktop/MAML-Edge/deploy_layer)
- [test_layer](/D:/Desktop/MAML-Edge/test_layer)
- [edge_layer](/D:/Desktop/MAML-Edge/edge_layer)
- [system_layer](/D:/Desktop/MAML-Edge/system_layer)

---

## 2. 全链路实验是怎么走的

### 第 1 层：数据层

原始数据来自 `CWRU / HST`。  
[data_layer](/D:/Desktop/MAML-Edge/data_layer) 负责：

- 读取不同域、不同故障标签的数据
- 生成 `FFT / STFT / WT` 三种输入表示
- 为训练和系统预测提供统一输入格式

训练时由：

- [preprocess_cwru.py](/D:/Desktop/MAML-Edge/data_layer/preprocess_cwru.py)
- [fault_datasets.py](/D:/Desktop/MAML-Edge/data_layer/fault_datasets.py)

参与数据构建。

### 第 2 层：模型层

[train.py](/D:/Desktop/MAML-Edge/train.py) 是统一入口。  
它会根据 `--algorithm` 分发到：

- [train_maml.py](/D:/Desktop/MAML-Edge/model_layer/train_maml.py)
- [train_protonet.py](/D:/Desktop/MAML-Edge/model_layer/train_protonet.py)
- [train_cnn.py](/D:/Desktop/MAML-Edge/model_layer/train_cnn.py)

训练完成后会输出：

- `*_best.pt`
- `*_history.json`

这些文件保存在：

- [checkpoints](/D:/Desktop/MAML-Edge/checkpoints)

其中 `history.json` 记录训练过程中的：

- `meta_train_acc`
- `meta_train_loss`
- `meta_test_acc`
- `meta_test_loss`

这部分是论文里“算法实验结果”的第一手来源。

### 第 3 层：部署层

训练得到最佳模型后，进入 [deploy_layer](/D:/Desktop/MAML-Edge/deploy_layer)：

- [compression.py](/D:/Desktop/MAML-Edge/deploy_layer/compression.py)
- [onnx_exporter.py](/D:/Desktop/MAML-Edge/deploy_layer/onnx_exporter.py)
- [runtime_backends.py](/D:/Desktop/MAML-Edge/deploy_layer/runtime_backends.py)
- [inference_service.py](/D:/Desktop/MAML-Edge/deploy_layer/inference_service.py)

这里完成：

- 结构化剪枝
- ONNX 导出
- INT8 量化
- 运行时后端封装

最终输出到：

- [deploy_artifacts](/D:/Desktop/MAML-Edge/deploy_artifacts)

每个实验目录下都有一个：

- `compression_summary.json`

它记录：

- 实验配置
- 最佳训练记录
- `float / int8` 部署指标
- 剪枝率、参数量、压缩比例
- 模型路径、原型路径、文件大小

这部分是论文里“部署压缩实验”和“模型轻量化对比”的核心数据来源。

### 第 4 层：测试层

[test_layer](/D:/Desktop/MAML-Edge/test_layer) 负责把训练和部署产物整理成论文可用结果：

- [benchmark.py](/D:/Desktop/MAML-Edge/test_layer/benchmark.py)
- [experiment_runner.py](/D:/Desktop/MAML-Edge/test_layer/experiment_runner.py)
- [result_aggregator.py](/D:/Desktop/MAML-Edge/test_layer/result_aggregator.py)

这层做三件事：

1. 读取单个 `compression_summary.json`
2. 导出单个实验的 benchmark 行
3. 聚合多个实验结果，输出 JSON / CSV

所以论文里的主结果表、部署对比表，应该优先从这层导出，而不是手工抄。

### 第 5 层：边缘层

[edge_layer](/D:/Desktop/MAML-Edge/edge_layer) 负责把信号送进系统：

- [simulator](/D:/Desktop/MAML-Edge/edge_layer/simulator)
- [mqtt_client](/D:/Desktop/MAML-Edge/edge_layer/mqtt_client)

当前已跑通的是：

- synthetic 模拟数据
- MQTT 发布
- 与系统层联调

`STM32 / ESP32` 目前仍是接口和说明预留，不应写成完整实机系统已经实现。

### 第 6 层：系统层

[system_layer](/D:/Desktop/MAML-Edge/system_layer) 把部署产物真正变成在线系统：

- [main.py](/D:/Desktop/MAML-Edge/system_layer/backend/main.py)
- [predictor.py](/D:/Desktop/MAML-Edge/system_layer/backend/predictor.py)
- [mqtt_worker.py](/D:/Desktop/MAML-Edge/system_layer/backend/mqtt_worker.py)
- [websocket_manager.py](/D:/Desktop/MAML-Edge/system_layer/backend/websocket_manager.py)
- [service_stats.py](/D:/Desktop/MAML-Edge/system_layer/backend/service_stats.py)

负责：

- `HTTP /predict`
- `MQTT` 消费
- `WebSocket` 实时推送
- `history / alerts` 存储
- `Vue + ECharts` 控制台
- 最小可用的 `/adapt` 原型更新

前端位于：

- [index.html](/D:/Desktop/MAML-Edge/system_layer/frontend/webui/index.html)
- [app.js](/D:/Desktop/MAML-Edge/system_layer/frontend/webui/app.js)
- [styles.css](/D:/Desktop/MAML-Edge/system_layer/frontend/webui/styles.css)

---

## 3. 各种统计指标是怎么获得的

### 3.1 训练指标

来源：

- [checkpoints](/D:/Desktop/MAML-Edge/checkpoints) 下的 `*_history.json`

产生方式：

- 训练循环每轮记录 `train/test accuracy` 与 `loss`
- 由各算法训练脚本自动写入

用于：

- 算法性能表
- 学习曲线
- 最佳模型选择

### 3.2 部署指标

来源：

- [deploy_artifacts](/D:/Desktop/MAML-Edge/deploy_artifacts) 下的 `compression_summary.json`

产生方式：

- 剪枝后重新评估
- ONNX Runtime 对 `float / int8` 模型做推理测试
- 结果写入 summary

典型字段：

- `accuracy`
- `avg_latency_ms`
- `baseline_params`
- `pruned_params`
- `parameter_reduction_ratio`
- `artifact_sizes_mb`

用于：

- 压缩前后对比
- `float vs int8` 对比
- 模型大小与时延分析

### 3.3 系统级在线指标

来源：

- `GET /system/stats`
- `GET /history`
- `GET /alerts`
- `WS /ws/realtime`

产生方式：

- [predictor.py](/D:/Desktop/MAML-Edge/system_layer/backend/predictor.py) 在每次预测时计算：
  - `preprocess_latency_ms`
  - `inference_latency_ms`
  - `end_to_end_latency_ms`
- [service_stats.py](/D:/Desktop/MAML-Edge/system_layer/backend/service_stats.py) 汇总：
  - direct 请求数
  - mqtt 请求数
  - 平均时延
  - 报警数
  - 适配次数

用于：

- 系统响应性能分析
- direct 与 mqtt 路径对比
- 端到端在线系统展示

### 3.4 前端实时展示

来源：

- `/health`
- `/model/info`
- `/benchmark/current`
- `/system/stats`
- `/history`
- `/alerts`
- `/ws/realtime`

产生方式：

- Vue 控制台轮询接口并接收 WebSocket 推送
- ECharts 绘制时延与置信度曲线

用于：

- 答辩演示
- 系统工作过程可视化

---

## 4. 本项目当前最推荐的实验路线

### 路线 A：算法实验

1. 训练 `MAML / ProtoNet / CNN`
2. 比较 `FFT / STFT / WT`
3. 比较 `1-shot / 5-shot`
4. 查看 `checkpoints/*_history.json`

### 路线 B：部署实验

1. 用最佳模型生成 `compression_summary.json`
2. 用 [benchmark.py](/D:/Desktop/MAML-Edge/test_layer/benchmark.py) 导出单个实验行
3. 用 [result_aggregator.py](/D:/Desktop/MAML-Edge/test_layer/result_aggregator.py) 聚合成论文表格

### 路线 C：系统实验

1. 启动 FastAPI
2. 启动 MQTT broker
3. 跑 Direct Predict
4. 跑 MQTT Simulation
5. 查看 `/history /alerts /system/stats`
6. 打开前端控制台和 WebSocket 实时监控

---

## 5. 常用命令

### 5.1 训练

```bash
python train.py --mode train --algorithm maml --dataset CWRU --preprocess STFT --shots 5 --ways 5
python train.py --mode train --algorithm protonet --dataset CWRU --preprocess STFT --shots 5 --ways 5 --enable_compression true
python train.py --mode train --algorithm cnn --dataset CWRU --preprocess FFT --shots 5 --ways 5
```

### 5.2 从已有最佳权重导出部署产物

```bash
python train.py --mode deploy --algorithm maml --dataset CWRU --preprocess STFT --shots 5 --ways 5
python train.py --mode deploy --algorithm protonet --dataset CWRU --preprocess STFT --shots 5 --ways 5
python train.py --mode deploy --algorithm cnn --dataset CWRU --preprocess FFT --shots 5 --ways 5
```

### 5.3 启动系统

Git Bash:

```bash
export MAML_EDGE_MODEL_SUMMARY_PATH="deploy_artifacts/<experiment_title>/compression_summary.json"
python -m uvicorn system_layer.backend.main:app --host 0.0.0.0 --port 8000
```

PowerShell:

```powershell
$env:MAML_EDGE_MODEL_SUMMARY_PATH="deploy_artifacts\<experiment_title>\compression_summary.json"
python -m uvicorn system_layer.backend.main:app --host 0.0.0.0 --port 8000
```

### 5.4 启动 MQTT

```bash
mosquitto -p 1883
```

### 5.5 导出 benchmark 行

```bash
python test_layer/benchmark.py \
  --summary_path deploy_artifacts/<experiment_title>/compression_summary.json \
  --rows_only \
  --output_format csv \
  --output_path logs/<experiment_title>_benchmark_row.csv
```

### 5.6 聚合所有部署结果

```bash
python test_layer/result_aggregator.py \
  --summary_glob "deploy_artifacts/*/compression_summary.json" \
  --output_format csv \
  --output_path logs/thesis_benchmark_rows.csv
```

### 5.7 批量实验矩阵

```bash
python test_layer/experiment_runner.py \
  --algorithms maml,protonet,cnn \
  --preprocesses FFT,STFT \
  --shots 1,5
```

执行一个小子集：

```bash
python test_layer/experiment_runner.py \
  --algorithms maml,protonet \
  --preprocesses STFT \
  --shots 1,5 \
  --iters 100 \
  --meta_batch_size 16 \
  --train_task_num 50 \
  --test_task_num 20 \
  --execute
```

---

## 6. 论文里如何表述更稳妥

建议明确区分三类内容：

### 已完成

- 六层一体化架构
- 少样本训练主线
- 剪枝 + ONNX + INT8 部署
- FastAPI + MQTT + WebSocket 在线系统
- Vue + ECharts 可视化控制台
- 最小可用的 ProtoNet runtime prototype update

### 已支持或已预留，但未充分实测

- `openvino`
- `tensorrt`

### 不应写成“完整实现”

- 复杂 `STM32 / ESP32` 实机工程
- 完整在线再训练系统

---

## 7. 一句话理解整个项目

这套代码不是单纯的训练仓库，也不是单纯的演示前端，而是一条从数据、训练、部署、测试、边缘接入到系统在线运行的完整本科毕设链路。  
论文里的算法结果、部署结果、系统结果，分别来自不同层，但最终都能在同一个项目里闭环验证。
