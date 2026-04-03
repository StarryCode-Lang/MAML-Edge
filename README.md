# MAML-Edge

工业物联网下基于元学习的边缘设备少样本故障诊断系统。

项目主结构保持为六层一体化：

- `data_layer`：CWRU / HST 数据读取、FFT / STFT / WT 预处理、few-shot 数据构造
- `model_layer`：`MAML`、`ProtoNet`、`CNN` 的训练与评估
- `deploy_layer`：结构化剪枝、恢复微调、ONNX 导出、INT8 PTQ、推理后端选择
- `test_layer`：benchmark、实验矩阵与结果聚合
- `edge_layer`：边缘侧信号模拟、轻量预处理、MQTT 上传
- `system_layer`：FastAPI、MQTT、WebSocket、历史/报警存储与 Vue + ECharts 控制台

统一训练入口：

```bash
python train.py --mode {train|deploy} --algorithm {maml|protonet|cnn} ...
```

部署层也可以单独运行：

```bash
python deploy_layer/deploy.py --algorithm {maml|protonet|cnn} ...
```

## 环境

推荐：

- `Python 3.9`
- 完整训练环境按 PyTorch 官方方式安装 `torch`

完整环境安装：

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

## 推理后端

- `onnxruntime`：默认主方案
- `tensorrt`：NVIDIA / Jetson 平台可选
- `openvino`：Intel 平台可选

其中 `onnxruntime` 已完成主流程验证；`tensorrt` 和 `openvino` 当前属于支持/预留后端，是否实测取决于目标平台。

## 完整实验链路

### 1. 数据层

`data_layer` 负责：

- 读取 `CWRU / HST` 原始数据
- 构造 `FFT / STFT / WT` 输入
- 为训练、部署与系统层保持一致的输入格式

核心文件：

- [preprocess_cwru.py](/D:/Desktop/MAML-Edge/data_layer/preprocess_cwru.py)
- [fault_datasets.py](/D:/Desktop/MAML-Edge/data_layer/fault_datasets.py)

### 2. 模型层

训练入口为 [train.py](/D:/Desktop/MAML-Edge/train.py)，会根据 `--algorithm` 分发到：

- [train_maml.py](/D:/Desktop/MAML-Edge/model_layer/train_maml.py)
- [train_protonet.py](/D:/Desktop/MAML-Edge/model_layer/train_protonet.py)
- [train_cnn.py](/D:/Desktop/MAML-Edge/model_layer/train_cnn.py)

训练后会在 `checkpoints/` 下生成：

- `*_best.pt`
- `*_history.json`

`*_history.json` 是训练指标的直接来源，主要包括：

- `meta_train_acc`
- `meta_train_loss`
- `meta_test_acc`
- `meta_test_loss`

### 3. 部署层

训练后的模型进入 `deploy_layer`，完成：

- 结构化剪枝
- 恢复微调
- ONNX 导出
- INT8 PTQ
- 推理后端封装

核心文件：

- [compression.py](/D:/Desktop/MAML-Edge/deploy_layer/compression.py)
- [onnx_exporter.py](/D:/Desktop/MAML-Edge/deploy_layer/onnx_exporter.py)
- [runtime_backends.py](/D:/Desktop/MAML-Edge/deploy_layer/runtime_backends.py)
- [inference_service.py](/D:/Desktop/MAML-Edge/deploy_layer/inference_service.py)

最终产物输出到：

```text
deploy_artifacts/<experiment_title>/
```

主要文件：

- `compression_summary.json`
- `*_float.onnx`
- `*_int8.onnx`
- `ProtoNet` 额外输出 `*_prototypes.npz`

`compression_summary.json` 记录：

- 实验描述
- 部署类型与后端
- 压缩前后模型路径与文件大小
- 剪枝率、参数量、压缩比例
- `float / int8` 推理准确率与时延

### 4. 测试层

`test_layer` 负责把训练与部署结果整理成论文可用数据。

核心文件：

- [benchmark.py](/D:/Desktop/MAML-Edge/test_layer/benchmark.py)
- [experiment_runner.py](/D:/Desktop/MAML-Edge/test_layer/experiment_runner.py)
- [result_aggregator.py](/D:/Desktop/MAML-Edge/test_layer/result_aggregator.py)

作用：

- 读取单个 `compression_summary.json`
- 导出单实验 benchmark 行
- 聚合多实验结果为 `JSON / CSV`
- 生成批量实验清单

### 5. 边缘层

`edge_layer` 负责把信号送入在线系统，当前已打通：

- synthetic 信号模拟
- MQTT 发布
- 与系统层联调

主要文件：

- [publish_signal.py](/D:/Desktop/MAML-Edge/edge_layer/simulator/publish_signal.py)
- [preprocess.py](/D:/Desktop/MAML-Edge/edge_layer/simulator/preprocess.py)
- [sample_payloads.py](/D:/Desktop/MAML-Edge/edge_layer/simulator/sample_payloads.py)
- [publisher.py](/D:/Desktop/MAML-Edge/edge_layer/mqtt_client/publisher.py)

说明：

- `--source synthetic` 只依赖 `requirements.edge-system.txt`
- `--source cwru` 需要完整训练环境
- `STM32 / ESP32` 当前主要是接入接口与说明，未作为复杂实机工程宣称完成

### 6. 系统层

`system_layer` 把部署产物真正接入在线服务。

核心文件：

- [main.py](/D:/Desktop/MAML-Edge/system_layer/backend/main.py)
- [predictor.py](/D:/Desktop/MAML-Edge/system_layer/backend/predictor.py)
- [mqtt_worker.py](/D:/Desktop/MAML-Edge/system_layer/backend/mqtt_worker.py)
- [websocket_manager.py](/D:/Desktop/MAML-Edge/system_layer/backend/websocket_manager.py)
- [service_stats.py](/D:/Desktop/MAML-Edge/system_layer/backend/service_stats.py)
- [history_store.py](/D:/Desktop/MAML-Edge/system_layer/storage/history_store.py)
- [alert_store.py](/D:/Desktop/MAML-Edge/system_layer/storage/alert_store.py)

系统层提供：

- `HTTP /predict`
- `MQTT` 在线消费
- `WebSocket` 实时推送
- `history / alerts` 存储
- `GET /system/stats` 系统统计
- 最小可用的 `/adapt` 原型更新
- Vue + ECharts 浏览器控制台

## 指标从哪里来

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

- 压缩前后对比
- `float vs int8` 对比
- 模型大小、参数量、平均推理时延分析

### 系统级在线指标

来源：

- `GET /system/stats`
- `GET /history`
- `GET /alerts`
- `WS /ws/realtime`

由 [predictor.py](/D:/Desktop/MAML-Edge/system_layer/backend/predictor.py) 和 [service_stats.py](/D:/Desktop/MAML-Edge/system_layer/backend/service_stats.py) 统计：

- `preprocess_latency_ms`
- `inference_latency_ms`
- `end_to_end_latency_ms`
- direct 请求数与平均时延
- mqtt 请求数与平均时延
- 报警次数
- 适配次数与支持样本数

### 时延字段语义

- `inference_latency_ms`：部署层纯推理时延
- `preprocess_latency_ms`：系统层在线预处理时延
- `end_to_end_latency_ms`：系统层在线端到端时延
- `latency_ms`：兼容字段，等于 `end_to_end_latency_ms`

## 从头训练并完成压缩导出

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
  --query_shots 5 \
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
  --query_shots 5 \
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

## 后两层联调

### edge_layer

推荐启动方式：

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

兼容直接脚本方式：

```bash
python edge_layer/simulator/publish_signal.py \
  --source synthetic \
  --device_id esp32-sim-01 \
  --host 127.0.0.1 \
  --port 1883 \
  --topic maml-edge/devices/esp32-sim-01/signal \
  --count 5 \
  --interval 1.0
```

### system_layer

推荐启动方式：

```bash
python -m uvicorn system_layer.backend.main:app --host 0.0.0.0 --port 8000
```

兼容直接命令：

```bash
uvicorn system_layer.backend.main:app --host 0.0.0.0 --port 8000
```

推荐显式指定模型：

```bash
MAML_EDGE_MODEL_SUMMARY_PATH=deploy_artifacts/<experiment_title>/compression_summary.json
```

只有在显式关闭严格模式时，系统才允许按最新产物自动发现：

```bash
MAML_EDGE_STRICT_MODEL_SELECTION=0
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

浏览器控制台：

```text
http://127.0.0.1:8000/
```

当前页面能力包括：

- 浏览和切换 `deploy_artifacts` 中的模型
- 查看 `/health`、`/model/info`、`/benchmark/current`、`/system/stats`
- 直接发起 `/predict`
- 从页面触发 synthetic / CWRU 模拟
- 执行最小可用的 `ProtoNet` 原型适配
- 清空并查看 `history / alerts`
- 通过 WebSocket 实时查看诊断结果

### 推荐联调顺序

1. 启动 `Mosquitto`
2. 启动 `system_layer` 后端
3. 用 `POST /predict` 验证服务推理
4. 运行 `edge_layer` 模拟器
5. 检查 `GET /history`
6. 检查 `GET /alerts`
7. 再做 `WS /ws/realtime` 联调
8. 若使用 `ProtoNet` 部署产物，再验证 `POST /adapt`

Windows 示例：

```bash
mosquitto -p 1883
set MAML_EDGE_MODEL_SUMMARY_PATH=deploy_artifacts\<experiment_title>\compression_summary.json
python -m uvicorn system_layer.backend.main:app --host 0.0.0.0 --port 8000
python -m edge_layer.simulator.publish_signal --source synthetic --count 5
```

## 测试与论文数据导出

单实验 benchmark：

```bash
python test_layer/benchmark.py \
  --summary_path deploy_artifacts/<experiment_title>/compression_summary.json
```

导出单实验论文行：

```bash
python test_layer/benchmark.py \
  --summary_path deploy_artifacts/<experiment_title>/compression_summary.json \
  --rows_only \
  --output_format csv \
  --output_path logs/<experiment_title>_benchmark_row.csv
```

聚合全部部署结果：

```bash
python test_layer/result_aggregator.py \
  --summary_glob "deploy_artifacts/*/compression_summary.json" \
  --output_format csv \
  --output_path logs/thesis_benchmark_rows.csv
```

生成批量实验矩阵：

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

说明：

- `benchmark.py` 读取的是部署层 `compression_summary.json`
- `avg_latency_ms` 表示部署层平均推理时延，不等同于系统层在线端到端时延
- 系统级在线指标应从 `/system/stats`、`/history`、`/alerts` 和 WebSocket 获取

## 论文表述建议

### 已完成

- 六层一体化结构
- 少样本训练主线
- 剪枝 + ONNX + INT8 部署
- FastAPI + MQTT + WebSocket 在线系统
- Vue + ECharts 浏览器控制台
- 最小可用的 ProtoNet runtime prototype update

### 已支持或已预留，但未充分实测

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
