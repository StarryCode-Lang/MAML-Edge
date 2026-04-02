# MAML-Edge

工业物联网下基于元学习的边缘设备少样本故障诊断系统。

当前项目结构保持为：

- `data_layer`：CWRU / HST 数据读取与 FFT、STFT、WT 预处理
- `model_layer`：`MAML`、`ProtoNet`、`CNN` 训练与评估
- `deploy_layer`：结构化剪枝、恢复微调、ONNX 导出、INT8 PTQ、推理后端选择
- `test_layer`：部署结果与指标检查
- `edge_layer`：边缘侧信号模拟、轻量预处理、MQTT 上传
- `system_layer`：FastAPI、MQTT、WebSocket、存储与前端占位

统一入口：

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
- GPU 环境按 PyTorch 官方方式安装 `torch`

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

## 推理后端

- `onnxruntime`：默认主方案
- `tensorrt`：NVIDIA / Jetson 平台可选
- `openvino`：Intel 平台可选

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

## deploy_artifacts 输出

```text
deploy_artifacts/<experiment_title>/
```

主要文件：

- `compression_summary.json`
- `*_float.onnx`
- `*_int8.onnx`
- `ProtoNet` 额外输出 `*_prototypes.npz`

## edge_layer

当前已提供：

- `edge_layer/simulator/publish_signal.py`
- `edge_layer/simulator/preprocess.py`
- `edge_layer/simulator/sample_payloads.py`
- `edge_layer/mqtt_client/publisher.py`

作用：

- Python 模拟边缘设备采集
- 计算 RMS、峰值、FFT 摘要
- 生成事件触发上传消息
- 通过 MQTT 发布原始片段与特征摘要

示例：

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

## system_layer

当前已提供：

- `system_layer/backend/main.py`
- `system_layer/backend/mqtt_worker.py`
- `system_layer/backend/predictor.py`
- `system_layer/backend/websocket_manager.py`
- `system_layer/storage/history_store.py`
- `system_layer/storage/alert_store.py`
- `system_layer/config/settings.py`

作用：

- FastAPI 服务入口
- MQTT 订阅与在线推理
- WebSocket 实时推送
- 历史记录与报警记录
- 从 `deploy_artifacts` 加载 ONNX 与 `ProtoNet` 原型

启动后端：

```bash
uvicorn system_layer.backend.main:app --host 0.0.0.0 --port 8000
```

主要接口：

- `GET /health`
- `GET /model/info`
- `GET /history`
- `GET /alerts`
- `POST /predict`
- `POST /adapt`
- `WS /ws/realtime`

前端目录当前为占位：

```text
system_layer/frontend/webui/
```

后续用于放置 `Vue + ECharts` 页面。

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
- [Prototypical Networks for Few-shot Learning](https://arxiv.org/pdf/1703.05175)
- [Matching Networks for One Shot Learning](https://arxiv.org/pdf/1606.04080)
- [CWRU official Database](https://engineering.case.edu/bearingdatacenter/download-data-file)

## License

MIT
