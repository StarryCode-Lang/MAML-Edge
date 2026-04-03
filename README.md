# MAML-Edge

面向本科毕业设计的工业故障诊断项目，覆盖数据处理、少样本训练、轻量化部署、边缘接入、系统服务与可视化控制台六层链路。

当前代码基线已经具备：

- `data_layer`：CWRU / HST 数据读取与 `FFT / STFT / WT` 预处理
- `model_layer`：`MAML / ProtoNet / CNN` 训练与统一实验配置
- `deploy_layer`：结构化剪枝、ONNX 导出、INT8 量化、多后端推理封装
- `test_layer`：部署基准检查、批量实验矩阵运行、结果聚合导出
- `edge_layer`：模拟器、MQTT 发布、真实硬件接口预留
- `system_layer`：FastAPI + MQTT + WebSocket + Vue + ECharts 一体化监控控制台

## 1. 项目结构

```text
MAML-Edge/
├─ data_layer/                  # 数据集读取与预处理
├─ model_layer/                 # MAML / ProtoNet / CNN 训练主线
├─ deploy_layer/                # 压缩、导出、ONNX / INT8 推理
├─ test_layer/                  # benchmark、实验矩阵、结果聚合
├─ edge_layer/                  # 边缘模拟与 MQTT 发布
├─ system_layer/                # 后端服务、存储、前端控制台
├─ deploy_artifacts/            # 部署产物
├─ checkpoints/                 # 训练权重与 history
├─ logs/                        # 训练与实验记录
└─ train.py                     # 统一 train / deploy 入口
```

## 2. 论文导向的新增能力

本分支在原有演示系统基础上，补了以下论文级功能：

### 2.1 部署摘要增强

`deploy_layer/compression.py` 现在会在 `compression_summary.json` 中补齐：

- 实验描述：算法、数据集、预处理、`ways / shots / query_shots`
- 域配置：训练域、测试域、故障标签
- 模型部署类型：`classifier / encoder_with_prototypes`
- 部署后端：`onnxruntime / openvino / tensorrt`
- 剪枝信息：剪枝率、参数量、参数压缩比例
- 文件体积：`float / int8 / prototype / checkpoint / history`
- 训练产物路径：`best_checkpoint_path / history_path`

这些字段可以直接支撑论文中的部署对比表。

### 2.2 Runtime Prototype Adapt

`system_layer/backend/main.py` 中的 `/adapt` 已从占位接口变为最小可用版本：

- 仅对 `ProtoNet` 这类 `encoder_with_prototypes` 部署开放
- 支持通过少量 `support_samples` 或 `support_features` 更新运行时 prototypes
- 默认策略是原型均值更新
- 适配只发生在系统运行时，不改训练主线，不做在线再训练

注意：

- 对 `MAML / CNN` 这类分类器部署，`/adapt` 会返回 `unsupported`
- 这是最小可用 few-shot 原型更新，不是完整在线学习框架

### 2.3 系统级统计

新增 `GET /system/stats`，用于统计：

- `direct` 请求数与平均 `preprocess / inference / end-to-end latency`
- `mqtt` 请求数与平均 `preprocess / inference / end-to-end latency`
- 报警触发次数
- 适配请求次数与样本数量

这部分指标用于论文中的系统性能分析章节。

### 2.4 批量实验与结果聚合

新增：

- `test_layer/experiment_runner.py`
- `test_layer/result_aggregator.py`

用途：

- 批量生成实验矩阵并可直接执行
- 从多个 `compression_summary.json` 聚合论文表格原始数据
- 输出 JSON / CSV，便于后续论文制表

## 3. 统一运行方式

### 3.1 训练

```bash
python train.py --mode train --algorithm maml --dataset CWRU --preprocess STFT --shots 5 --ways 5
python train.py --mode train --algorithm protonet --dataset CWRU --preprocess FFT --shots 5 --ways 5
python train.py --mode train --algorithm cnn --dataset CWRU --preprocess WT --shots 5 --ways 5
```

如需训练后直接生成部署产物：

```bash
python train.py --mode train --algorithm protonet --dataset CWRU --preprocess STFT --shots 5 --ways 5 --enable_compression true
```

### 3.2 从已有 checkpoint 生成部署产物

```bash
python train.py --mode deploy --algorithm maml --dataset CWRU --preprocess STFT --shots 5 --ways 5
python train.py --mode deploy --algorithm protonet --dataset CWRU --preprocess FFT --shots 5 --ways 5
python train.py --mode deploy --algorithm cnn --dataset CWRU --preprocess WT --shots 5 --ways 5
```

### 3.3 启动系统服务

#### Git Bash

```bash
export MAML_EDGE_MODEL_SUMMARY_PATH="deploy_artifacts/<experiment_title>/compression_summary.json"
python -m uvicorn system_layer.backend.main:app --host 0.0.0.0 --port 8000
```

#### PowerShell

```powershell
$env:MAML_EDGE_MODEL_SUMMARY_PATH="deploy_artifacts\<experiment_title>\compression_summary.json"
python -m uvicorn system_layer.backend.main:app --host 0.0.0.0 --port 8000
```

启动后：

- 首页控制台：`http://127.0.0.1:8000/`
- 健康检查：`GET /health`
- 当前模型：`GET /model/info`
- 系统统计：`GET /system/stats`

### 3.4 MQTT 联调

先启动 broker：

```bash
mosquitto -p 1883
```

再运行模拟器：

```bash
python -m edge_layer.simulator.publish_signal --source synthetic --count 5 --interval 1.0
```

或直接在 Web 控制台中触发 simulation。

## 4. API 概览

### 4.1 核心接口

- `GET /health`
- `GET /model/info`
- `GET /artifacts/summaries`
- `POST /model/select`
- `GET /benchmark/current`
- `GET /system/stats`
- `GET /history`
- `GET /alerts`
- `POST /storage/reset`
- `POST /predict`
- `POST /simulate/publish`
- `POST /adapt`
- `WS /ws/realtime`

### 4.2 `/adapt` 输入示例

#### 原始信号方式

```json
{
  "blend_factor": 0.5,
  "support_samples": [
    {
      "device_id": "support-01",
      "label": 0,
      "raw_signal": [0.01, 0.03, 0.02, 0.15, 0.22, 0.18, 0.03, 0.02]
    },
    {
      "device_id": "support-02",
      "label": 0,
      "raw_signal": [0.01, 0.03, 0.02, 0.15, 0.22, 0.18, 0.03, 0.02]
    }
  ]
}
```

#### 特征向量方式

```json
{
  "support_features": [
    {
      "label": 0,
      "embedding": [0.12, 0.03, -0.44, 0.81]
    }
  ]
}
```

返回结果会包含：

- `status`
- `message`
- `adaptation.updated_labels`
- `adaptation.sample_count`
- `adaptation.prototype_count`
- `model_info`

## 5. Benchmark 与论文表格导出

### 5.1 单个部署摘要检查

```bash
python test_layer/benchmark.py ^
  --summary_path deploy_artifacts/<experiment_title>/compression_summary.json ^
  --output_format json
```

导出单行 CSV：

```bash
python test_layer/benchmark.py ^
  --summary_path deploy_artifacts/<experiment_title>/compression_summary.json ^
  --rows_only ^
  --output_format csv ^
  --output_path logs/<experiment_title>_benchmark_row.csv
```

### 5.2 聚合所有部署结果

```bash
python test_layer/result_aggregator.py ^
  --summary_glob "deploy_artifacts/*/compression_summary.json" ^
  --output_format csv ^
  --output_path logs/thesis_benchmark_rows.csv
```

### 5.3 批量实验矩阵

只生成计划，不执行：

```bash
python test_layer/experiment_runner.py ^
  --algorithms maml,protonet,cnn ^
  --preprocesses FFT,STFT ^
  --shots 1,5 ^
  --test_domains 3
```

直接执行一个子矩阵：

```bash
python test_layer/experiment_runner.py ^
  --algorithms maml,protonet ^
  --preprocesses STFT ^
  --shots 1,5 ^
  --iters 100 ^
  --meta_batch_size 16 ^
  --train_task_num 50 ^
  --test_task_num 20 ^
  --execute
```

执行后会在 `logs/thesis_runs/latest/` 下输出：

- `experiment_manifest.json`
- `benchmark_rows.json`（若 summary 已生成）

## 6. Web 控制台说明

当前控制台位于 `system_layer/frontend/webui/`，由 `FastAPI` 直接托管。

能力包括：

- 模型切换
- 手工预测
- Prototype Adapt
- Synthetic / CWRU 模拟
- Benchmark Snapshot
- Runtime Stats
- History / Alerts
- WebSocket 实时监控
- Snapshot 复制与导出

前端保持：

- `Vue 3`
- `ECharts`
- 单后端启动

不需要额外起前端 dev server。

## 7. 论文表述建议

建议在论文中这样表述边界：

- 已完成：
  - 六层系统架构
  - 元学习 / 基线模型训练
  - 剪枝 + ONNX + INT8 部署
  - MQTT / WebSocket / 历史报警系统
  - Prototype runtime adaptation 最小可用实现
- 支持但未充分实测：
  - `openvino`
  - `tensorrt`
- 已准备接口或方案，但不应写成“完整实现”：
  - `STM32 / ESP32` 实机复杂工程
  - 完整在线再训练

## 8. 当前最推荐的论文产出流程

1. 跑训练与部署，生成多个 `compression_summary.json`
2. 用 `benchmark.py` 和 `result_aggregator.py` 导出论文表格原始数据
3. 启动系统服务，演示：
   - `/predict`
   - `/simulate/publish`
   - `/adapt`
   - `/history`
   - `/alerts`
   - `/ws/realtime`
4. 论文中分别写：
   - 算法实验结果
   - 部署压缩结果
   - 系统性能与演示验证

## 9. 说明

本仓库当前目标是：在本科毕设标准下，将“少样本故障诊断 + 轻量化部署 + 边缘协同实时监测”做成可训练、可部署、可演示、可量化分析的一体化项目。
