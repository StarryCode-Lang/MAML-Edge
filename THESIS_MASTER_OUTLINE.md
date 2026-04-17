# 本科毕业论文最终总纲

## 0. 文件定位

本文件是当前论文写作的**唯一主文件**。  
后续大纲调整、正文撰写、图表安排、结果映射，统一以本文件为准。

本文件的依据包括：

- 学校与学院格式文件 PDF 原文：
  - `D:/Desktop/附件Z-2：吉林大学本科毕业论文（设计）撰写要求与书写格式.pdf`
  - `D:/Desktop/附件Z-3：学院本科毕业论文（设计）撰写参考模板.pdf`
  - `D:/Desktop/附件Z-4：优秀毕业论文（设计）的缩写格式.pdf`
- 开题报告：
  - `D:/Desktop/开题报告.pdf`
- 两份参考大纲：
  - `D:/Desktop/MAML_Edge_论文大纲.md`
  - `D:/Desktop/MAML_Edge_论文大纲 - 副本.md`
- 当前项目代码与实验结果：
  - 分支：`codex/thesis-upgrade`
  - 代码、结果、论文表格均以当前分支实际内容为准

取舍优先级：

1. 当前分支代码与最终结果
2. 学校 PDF 原始格式要求
3. 开题报告中的研究方向与动机
4. 两份 Markdown 大纲中的可复用写法

## 1. 论文总体定位

### 1.1 最终定位

本论文最合适的定位是：

**面向边缘部署的少样本轴承故障诊断方法研究与系统实现**

这是一个“方法 + 部署 + 系统验证”的应用研究型本科论文，不是纯算法论文，也不是纯系统开发文档。

### 1.2 不能写偏的三条边界

#### 1.2.1 不能写成纯 MAML 论文

项目名虽然是 `MAML-Edge`，但当前实验结果表明：

- `ProtoNet` 综合表现最好
- `MAML` 次之
- `CNN` 为对照基线

因此正文里不能把 `MAML` 写成全文唯一主角，更不能写成最终全局最优模型。

#### 1.2.2 不能写成纯系统工程论文

你的 strongest evidence 不是 UI 页面或接口数，而是：

- 27 组主实验
- 多 seed 稳定性
- 多目标域跨工况泛化
- 压缩消融
- direct 通道系统闭环 benchmark

因此系统是支撑章节，不是唯一主线。

#### 1.2.3 不能把开题里未完成的内容写成已完成结果

开题中提到过但当前主结果没有完整落地的内容，包括：

- `Paderborn`
- 知识蒸馏主实验
- Jetson Nano / Atlas 200I 主部署结果
- TensorRT 主结果
- 完整 direct vs MQTT 系统对比

正文主实验必须以当前分支实际结果为准。

## 2. 学校格式硬要求

## 2.1 结构要求

根据 `附件Z-2`，论文基本结构必须包含：

- 前置部分
  - 封面
  - 序或前言（可选）
  - 摘要及关键词
  - 目录
  - 插图和附表清单（可选）
  - 符号、标志、缩略词、单位、术语注释表（可选）
- 主体部分
  - 引言（或绪论）
  - 正文
  - 结论
  - 参考文献
  - 注释（可选）
  - 附录（可选）
- 结尾部分
  - 致谢
  - 封底

## 2.2 标题、摘要、关键词

- 中文题目不超过 `30` 个汉字
- 中文摘要 `300-500` 字为宜
- 关键词一般 `3-5` 个
- 外文摘要必须对应中文摘要

## 2.3 页码与章节

- 主体部分从引言开始
- **每章另起一页**
- 前置部分页码用罗马数字
- 正文从引言开始用阿拉伯数字连续编号

## 2.4 排版要求

按 `附件Z-3` 学院模板，推荐采用：

- 页面设置
  - 左右：`3 cm`
  - 上下：`2 cm`
  - 页眉页脚：`1.5 cm`
- 字号字体
  - 中文题目：宋体小三，居中
  - 中文摘要标题：宋体四号加粗，居中
  - 中文摘要正文：宋体小四，`1.5` 倍行距
  - 英文题目：Times New Roman 小三，居中
  - 英文摘要标题：Times New Roman 四号加粗，居中
  - 英文摘要正文：Times New Roman 小四，`1.5` 倍行距
  - 章标题：宋体三号，居中
  - 二级标题：宋体四号，居左
  - 三级标题：宋体小四，居左
  - 正文：宋体小四，`1.5` 倍行距

## 2.5 图、表、公式、参考文献

- 图、表、公式按章节编号
  - 图 `1-1`
  - 表 `2-2`
  - 式 `3-1`
- 图题在图下，表题在表上
- 图表标题：黑体五号
- 图内表内文字：宋体五号
- 公式居中，编号右对齐
- 参考文献按 `GB/T 7714-2015`
- 正文建议采用顺序编码制

## 3. 最终题目建议

优先推荐：

**面向边缘部署的少样本轴承故障诊断方法研究与系统实现**

备选：

- 基于元学习的少样本轴承故障诊断方法与边缘部署研究
- 面向工业物联网的少样本故障诊断与边缘推理系统设计

## 4. 全文结构总览

- 中文摘要
- Abstract
- 目录
- 图目录
- 表目录
- 符号说明表（建议）
- 第 1 章 绪论
- 第 2 章 相关理论与关键技术基础
- 第 3 章 系统总体设计
- 第 4 章 少样本故障诊断与边缘部署方法设计
- 第 5 章 实验设计与结果分析
- 第 6 章 系统实现与在线验证
- 第 7 章 总结与展望
- 参考文献
- 附录 A 代码仓库说明
- 附录 B 补充实验结果
- 致谢

正文建议总字数：

- `22000-28000`

## 5. 正文章节详细安排

## 第 1 章 绪论

### 章节目标

解决三个问题：

1. 为什么要做这个题
2. 别人做到了什么
3. 本文实际做了什么

### 字数建议

- `2500-3200`

### 章节结构

#### 1.1 研究背景与意义

字数建议：

- `700-900`

写作内容：

- 工业设备故障诊断与预测性维护背景
- 轴承故障诊断的重要性
- 少样本、跨工况、边缘实时性约束
- 本文研究意义

非文本内容：

- 图 `1-1`：论文技术路线图

#### 1.2 国内外研究现状

字数建议：

- `1000-1300`

建议分 4 段：

- 深度学习故障诊断
- 少样本故障诊断
- 元学习方法
- 边缘部署与模型压缩

写作要求：

- 每段都写“已有工作 -> 局限 -> 本文切入点”

可选非文本：

- 图 `1-2`：研究脉络或本文定位图

#### 1.3 研究问题与研究目标

字数建议：

- `350-500`

建议写成 4 点：

1. 少样本条件下的轴承故障诊断
2. 不同预处理和模型的效果比较
3. 压缩部署的精度与效率平衡
4. 在线系统闭环验证

#### 1.4 主要工作与贡献

字数建议：

- `350-500`

建议按 4 点写：

1. 构建 27 组主实验矩阵
2. 引入多 seed 与多域泛化验证
3. 设计压缩部署流程
4. 完成 direct 通道系统闭环验证

#### 1.5 论文结构安排

字数建议：

- `250-350`

非文本内容：

- 表 `1-1`：章节与任务对照表

### 本章写作边界

- 不要提前写细节实验数据
- 不要提前写系统接口实现

## 第 2 章 相关理论与关键技术基础

### 章节目标

给后续方法与实验提供理论铺垫，不重复教科书，不堆文献。

### 字数建议

- `3000-4200`

### 章节结构

#### 2.1 故障诊断与振动信号基础

字数建议：

- `500-700`

内容：

- 轴承故障诊断基本原理
- 振动信号在状态识别中的作用
- 工况变化引起的分布偏移

非文本内容：

- 图 `2-1`：故障/信号示意图

#### 2.2 信号预处理方法

字数建议：

- `900-1200`

##### 2.2.1 FFT

- 字数：`250-300`
- 公式：式 `2-1`

##### 2.2.2 STFT

- 字数：`300-400`
- 公式：式 `2-2`

##### 2.2.3 WT

- 字数：`300-400`
- 公式：式 `2-3`

非文本内容：

- 图 `2-2`：FFT/STFT/WT 对比示意图

#### 2.3 少样本学习与元学习基础

字数建议：

- `1000-1300`

##### 2.3.1 Few-shot 任务定义

- 字数：`200-250`
- 公式：式 `2-4`

##### 2.3.2 CNN 基线

- 字数：`180-250`
- 公式：式 `2-5`

##### 2.3.3 MAML

- 字数：`300-400`
- 公式：式 `2-6`、式 `2-7`
- 伪代码：算法 `1`

##### 2.3.4 ProtoNet

- 字数：`300-400`
- 公式：式 `2-8`、式 `2-9`
- 伪代码：算法 `2`

非文本内容：

- 图 `2-3`：few-shot episode 示意图
- 图 `2-4`：MAML 流程图
- 图 `2-5`：ProtoNet 示意图

#### 2.4 模型压缩与边缘推理基础

- 字数：`500-700`
- 公式：式 `2-10`、式 `2-11`
- 非文本：图 `2-6` 压缩部署概念图

#### 2.5 本章小结

- 字数：`120-180`

## 第 3 章 系统总体设计

### 章节目标

说明系统是如何从代码层面支撑论文实验与结果产出的。

### 字数建议

- `2200-3000`

### 章节结构

#### 3.1 系统需求分析

- 字数：`350-500`
- 内容：功能需求、性能需求、工程需求

#### 3.2 总体架构设计

- 字数：`500-700`
- 内容：六层结构
  - `data_layer`
  - `model_layer`
  - `deploy_layer`
  - `test_layer`
  - `edge_layer`
  - `system_layer`
- 非文本：图 `3-1` 系统总体架构图

#### 3.3 数据流与控制流设计

- 字数：`350-500`
- 非文本：
  - 图 `3-2` 训练-部署-出表流程图
  - 图 `3-3` 目录结构与产物关系图

#### 3.4 模块职责划分

- 字数：`500-700`
- 非文本：
  - 表 `3-1` 各模块职责表
  - 表 `3-2` 核心数据文件与用途映射表

必须对照的代码文件：

- [train.py](D:/Desktop/MAML-Edge/train.py)
- [data_layer/fault_datasets.py](D:/Desktop/MAML-Edge/data_layer/fault_datasets.py)
- [model_layer/train_cnn.py](D:/Desktop/MAML-Edge/model_layer/train_cnn.py)
- [model_layer/train_maml.py](D:/Desktop/MAML-Edge/model_layer/train_maml.py)
- [model_layer/train_protonet.py](D:/Desktop/MAML-Edge/model_layer/train_protonet.py)
- [deploy_layer/compression.py](D:/Desktop/MAML-Edge/deploy_layer/compression.py)
- [system_layer/backend/main.py](D:/Desktop/MAML-Edge/system_layer/backend/main.py)
- [test_layer/thesis_tables.py](D:/Desktop/MAML-Edge/test_layer/thesis_tables.py)
- [test_layer/paper_tables.py](D:/Desktop/MAML-Edge/test_layer/paper_tables.py)

#### 3.5 论文实验导表链路设计

- 字数：`300-450`
- 内容：
  - 为什么不手抄原始日志
  - 为什么论文主表以 `logs/thesis_tables/...` 为准
  - 为什么 `compression_summary.json` 是原始证据

#### 3.6 本章小结

- 字数：`120-180`

## 第 4 章 少样本故障诊断与边缘部署方法设计

### 章节目标

把“方法”写完整：问题定义、任务构造、模型设计、压缩部署、评价指标。

### 字数建议

- `2600-3400`

### 章节结构

#### 4.1 问题定义

- 字数：`250-350`
- 内容：
  - `5-way K-shot`
  - 跨工况
  - 边缘部署目标

#### 4.2 数据组织与任务构造

- 字数：`500-700`
- 内容：
  - 主数据集 `CWRU`
  - `012 -> 3`
  - `013 -> 2`
  - `023 -> 1`
  - `123 -> 0`
- 非文本：
  - 图 `4-1` 数据集与工况划分图
  - 表 `4-1` 数据集与标签设置表

#### 4.3 预处理与输入表示设计

- 字数：`500-700`
- 内容：
  - FFT 输入
  - STFT 输入
  - WT 输入
  - 为什么三者进入统一受控实验

#### 4.4 模型设计与对比策略

- 字数：`700-900`
- 内容：
  - CNN 基线
  - MAML
  - ProtoNet
- 非文本：
  - 图 `4-2` 统一方法框架图

#### 4.5 模型压缩与部署方法设计

- 字数：`450-650`
- 内容：
  - baseline
  - prune only
  - recovery
  - int8
- 非文本：
  - 图 `4-3` 压缩部署流程图
  - 算法 `3` 或 `4`：压缩部署流程伪代码

#### 4.6 评价指标设计

- 字数：`250-400`
- 公式：
  - 式 `4-1` Accuracy
  - 式 `4-2` Mean
  - 式 `4-3` Std
  - 式 `4-4` Compression ratio
- 非文本：
  - 表 `4-2` 实验变量与控制变量表
  - 表 `4-3` 评价指标定义表

#### 4.7 本章小结

- 字数：`120-180`

## 第 5 章 实验设计与结果分析

### 章节目标

这是全文最核心章节，用实验结果把方法、稳定性、泛化性和部署效果全部讲清楚。

### 字数建议

- `5000-7000`

### 章节结构

#### 5.1 实验环境与配置

- 字数：`400-600`
- 非文本：
  - 表 `5-1` 实验环境配置表
  - 表 `5-2` 主实验矩阵说明表

#### 5.2 主实验结果分析（27 组）

- 字数：`1400-1800`
- 主数据文件：
  - [thesis_tables.json](D:/Desktop/MAML-Edge/logs/thesis_tables/controlled/thesis_tables.json)
  - [thesis_tables.md](D:/Desktop/MAML-Edge/logs/thesis_tables/controlled/thesis_tables.md)
  - [benchmark_rows.csv](D:/Desktop/MAML-Edge/logs/thesis_tables/controlled/benchmark_rows.csv)
- 图表：
  - 表 `5-3`：27 组主实验总表
  - 图 `5-1`：准确率柱状图
  - 图 `5-2`：shot 折线图
  - 图 `5-3`：热力图
- 分析顺序：
  1. 预处理比较
  2. 模型比较
  3. shot 影响
  4. 最优组合总结

#### 5.3 多 seed 稳定性分析

- 字数：`900-1200`
- 数据文件：
  - [paper_balanced_report.md](D:/Desktop/MAML-Edge/logs/thesis_tables/paper_balanced/paper_balanced_report.md)
  - [table0_preprocess_model_matrix_mean_std.csv](D:/Desktop/MAML-Edge/logs/thesis_tables/paper_balanced/table0_preprocess_model_matrix_mean_std.csv)
  - [table1_model_performance_mean_std.csv](D:/Desktop/MAML-Edge/logs/thesis_tables/paper_balanced/table1_model_performance_mean_std.csv)
  - [table2_few_shot_mean_std.csv](D:/Desktop/MAML-Edge/logs/thesis_tables/paper_balanced/table2_few_shot_mean_std.csv)
- 图表：
  - 表 `5-4`：预处理 × 模型 `mean ± std`
  - 表 `5-5`：模型比较 `mean ± std`
  - 表 `5-6`：few-shot `mean ± std`
  - 图 `5-4`：误差棒图
  - 图 `5-5`：few-shot 稳定性图

#### 5.4 跨域泛化分析

- 字数：`900-1200`
- 数据文件：
  - [table3_domain_robustness.csv](D:/Desktop/MAML-Edge/logs/thesis_tables/paper_balanced/table3_domain_robustness.csv)
  - [source013_target2 benchmark_rows.csv](D:/Desktop/MAML-Edge/logs/thesis_tables/paper_balanced/domain/source013_target2/benchmark_rows.csv)
  - [source023_target1 benchmark_rows.csv](D:/Desktop/MAML-Edge/logs/thesis_tables/paper_balanced/domain/source023_target1/benchmark_rows.csv)
  - [source123_target0 benchmark_rows.csv](D:/Desktop/MAML-Edge/logs/thesis_tables/paper_balanced/domain/source123_target0/benchmark_rows.csv)
- 图表：
  - 表 `5-7`：跨域泛化结果表
  - 图 `5-6`：不同 target split 柱状图
  - 图 `5-7`：难度排序图

#### 5.5 压缩消融实验分析

- 字数：`900-1200`
- 数据文件：
  - [table4_compression_ablation.csv](D:/Desktop/MAML-Edge/logs/thesis_tables/paper_balanced/table4_compression_ablation.csv)
  - `deploy_artifacts/paper_balanced/ablation/.../compression_summary.json`
- 图表：
  - 表 `5-8`：压缩消融结果表
  - 图 `5-8`：精度-模型大小权衡图
  - 图 `5-9`：精度-时延散点图
  - 图 `5-10`：压缩阶段对比图

#### 5.6 结果讨论与可信度分析

- 字数：`600-900`
- 必写内容：
  1. 项目名与最优模型不一致
  2. 当前部署评估是 `random_per_episode`
  3. 但 `deployment_eval_episode_count = 1`
  4. 因此部署评估统计强度有限
  5. system benchmark 当前只有 direct
- 非文本：
  - 表 `5-9`：结论与限制对照表

#### 5.7 本章小结

- 字数：`150-220`

## 第 6 章 系统实现与在线验证

### 章节目标

证明你不是只做了离线模型比较，而是完成了可运行系统的在线验证。

### 字数建议

- `2200-3000`

### 章节结构

#### 6.1 系统实现方案

- 字数：`450-650`
- 必写：
  - `FastAPI`
  - `WebSocket`
  - `MQTT`
  - `Vue + ECharts`

#### 6.2 在线推理流程

- 字数：`300-450`
- 非文本：
  - 图 `6-1`：在线推理流程时序图

#### 6.3 系统性能验证

- 字数：`500-700`
- 数据文件：
  - [system_benchmark.json](D:/Desktop/MAML-Edge/logs/thesis_tables/paper_balanced/system_benchmark.json)
  - [table5_system_performance.csv](D:/Desktop/MAML-Edge/logs/thesis_tables/paper_balanced/table5_system_performance.csv)
- 图表：
  - 表 `6-1`：系统性能结果表
  - 图 `6-2`：系统时延拆解图
- 写作边界：
  - 只能写 direct benchmark
  - 不能写 direct 与 mqtt 对比已完成

#### 6.4 系统界面与功能展示

- 字数：`300-450`
- 图片：
  - 图 `6-3`：实时监控界面
  - 图 `6-4`：结果或历史界面

#### 6.5 本章小结

- 字数：`120-180`

## 第 7 章 总结与展望

### 章节目标

把全文结论、边界和后续方向收口。

### 字数建议

- `1500-2200`

### 章节结构

#### 7.1 全文工作总结

- 字数：`400-600`

#### 7.2 主要结论

- 字数：`500-700`
- 要求：每条结论至少对应一个结果依据

建议 6 条结论：

1. `WT`、`STFT` 精度整体高于 `FFT`
2. `ProtoNet` 综合最好，`MAML` 次之，`CNN` 为基线
3. 多 seed 结果表明 `ProtoNet` 最稳定
4. 跨域 split 表明方法具有一定工况泛化能力
5. 压缩显著减小模型规模，但 INT8 不保证一定更快
6. direct 通道系统闭环已完成

#### 7.3 存在的不足

- 字数：`250-400`
- 必写：
  - 部署评估统计强度不足
  - 仅 direct benchmark
  - 缺少更多真实设备和数据集

#### 7.4 未来工作展望

- 字数：`250-400`

## 6. 参考文献策略

### 6.1 格式

- 使用 `GB/T 7714-2015`
- 采用顺序编码制

### 6.2 数量建议

- `30-40` 篇最稳

### 6.3 分组建议

- 元学习基础论文：`3-5`
- 故障诊断相关：`8-12`
- 信号处理相关：`5-8`
- 压缩与边缘部署：`6-10`
- 工业物联网/系统背景：`3-5`

## 7. 附录安排

## 附录 A 代码仓库极简说明

控制在 `1` 页左右，说明：

- 项目名称：`MAML-Edge`
- 论文分支：`codex/thesis-upgrade`
- 顶层目录作用：
  - `data_layer`
  - `model_layer`
  - `deploy_layer`
  - `edge_layer`
  - `system_layer`
  - `test_layer`
  - `train.py`

## 附录 B 补充实验结果

建议放：

- 各 split 的原始 benchmark rows
- 代表性训练曲线
- 代表性 compression summary 关键字段

## 8. 全文图、表、公式、伪代码总编号规划

## 8.1 图

- 图 `1-1` 论文技术路线图
- 图 `1-2` 研究定位图
- 图 `2-1` 故障/信号示意图
- 图 `2-2` FFT/STFT/WT 对比图
- 图 `2-3` few-shot 任务构造图
- 图 `2-4` MAML 流程图
- 图 `2-5` ProtoNet 示意图
- 图 `2-6` 压缩部署概念图
- 图 `3-1` 系统总体架构图
- 图 `3-2` 训练-部署-出表流程图
- 图 `3-3` 目录与产物关系图
- 图 `4-1` 数据集与工况划分图
- 图 `4-2` 方法总框架图
- 图 `4-3` 压缩部署流程图
- 图 `5-1` 主实验柱状图
- 图 `5-2` shot 折线图
- 图 `5-3` 主实验热力图
- 图 `5-4` 稳定性误差棒图
- 图 `5-5` few-shot 稳定性图
- 图 `5-6` 跨域泛化柱状图
- 图 `5-7` 跨域难度排序图
- 图 `5-8` 精度-模型大小权衡图
- 图 `5-9` 精度-时延散点图
- 图 `5-10` 压缩阶段对比图
- 图 `6-1` 在线推理时序图
- 图 `6-2` 系统时延拆解图
- 图 `6-3` 实时监控界面
- 图 `6-4` 历史/结果界面

## 8.2 表

- 表 `1-1` 章节与任务对照表
- 表 `3-1` 各模块职责表
- 表 `3-2` 核心数据文件映射表
- 表 `4-1` 数据集与标签设置表
- 表 `4-2` 实验变量与控制变量表
- 表 `4-3` 评价指标定义表
- 表 `5-1` 实验环境配置表
- 表 `5-2` 主实验矩阵说明表
- 表 `5-3` 27 组主实验总表
- 表 `5-4` 稳定性主矩阵
- 表 `5-5` 模型比较 `mean ± std`
- 表 `5-6` few-shot `mean ± std`
- 表 `5-7` 跨域泛化结果表
- 表 `5-8` 压缩消融结果表
- 表 `5-9` 结论与限制对照表
- 表 `6-1` 系统性能结果表

## 8.3 公式

- 式 `2-1` FFT
- 式 `2-2` STFT
- 式 `2-3` WT
- 式 `2-4` Few-shot 任务定义
- 式 `2-5` 交叉熵损失
- 式 `2-6` MAML 内循环
- 式 `2-7` MAML 外循环
- 式 `2-8` ProtoNet 原型
- 式 `2-9` ProtoNet 分类概率
- 式 `2-10` 压缩率
- 式 `2-11` 平均时延
- 式 `4-1` Accuracy
- 式 `4-2` Mean
- 式 `4-3` Std
- 式 `4-4` Compression ratio

## 8.4 伪代码

- 算法 `1` MAML 训练流程
- 算法 `2` ProtoNet 推理流程
- 算法 `3` 压缩部署流程

## 9. 实验结果与正文的文件映射

## 9.1 主实验

- [thesis_tables.json](D:/Desktop/MAML-Edge/logs/thesis_tables/controlled/thesis_tables.json)
- [thesis_tables.md](D:/Desktop/MAML-Edge/logs/thesis_tables/controlled/thesis_tables.md)
- [benchmark_rows.csv](D:/Desktop/MAML-Edge/logs/thesis_tables/controlled/benchmark_rows.csv)

主要用于：

- 第 `5.2` 节主实验结果分析

## 9.2 多 seed

- [paper_balanced_report.md](D:/Desktop/MAML-Edge/logs/thesis_tables/paper_balanced/paper_balanced_report.md)
- [table0_preprocess_model_matrix_mean_std.csv](D:/Desktop/MAML-Edge/logs/thesis_tables/paper_balanced/table0_preprocess_model_matrix_mean_std.csv)
- [table1_model_performance_mean_std.csv](D:/Desktop/MAML-Edge/logs/thesis_tables/paper_balanced/table1_model_performance_mean_std.csv)
- [table2_few_shot_mean_std.csv](D:/Desktop/MAML-Edge/logs/thesis_tables/paper_balanced/table2_few_shot_mean_std.csv)

主要用于：

- 第 `5.3` 节稳定性分析

## 9.3 跨域泛化

- [table3_domain_robustness.csv](D:/Desktop/MAML-Edge/logs/thesis_tables/paper_balanced/table3_domain_robustness.csv)
- [source013_target2 benchmark_rows.csv](D:/Desktop/MAML-Edge/logs/thesis_tables/paper_balanced/domain/source013_target2/benchmark_rows.csv)
- [source023_target1 benchmark_rows.csv](D:/Desktop/MAML-Edge/logs/thesis_tables/paper_balanced/domain/source023_target1/benchmark_rows.csv)
- [source123_target0 benchmark_rows.csv](D:/Desktop/MAML-Edge/logs/thesis_tables/paper_balanced/domain/source123_target0/benchmark_rows.csv)

主要用于：

- 第 `5.4` 节跨域泛化分析

## 9.4 压缩消融

- [table4_compression_ablation.csv](D:/Desktop/MAML-Edge/logs/thesis_tables/paper_balanced/table4_compression_ablation.csv)
- `deploy_artifacts/paper_balanced/ablation/.../compression_summary.json`

主要用于：

- 第 `5.5` 节压缩消融分析

## 9.5 系统层

- [system_benchmark.json](D:/Desktop/MAML-Edge/logs/thesis_tables/paper_balanced/system_benchmark.json)
- [table5_system_performance.csv](D:/Desktop/MAML-Edge/logs/thesis_tables/paper_balanced/table5_system_performance.csv)

主要用于：

- 第 `6.3` 节系统性能验证

## 10. 全文必须坚持的事实边界

1. 当前综合最好模型是 `ProtoNet`
2. `MAML` 是重要对比对象和主分析 profile，不是全局最优
3. `FFT` 主要优势在低时延，不在最高精度
4. 压缩显著减小模型大小，但 INT8 不保证一定更快
5. 当前系统 benchmark 只有 `direct`
6. 当前部署评估 `deployment_eval_episode_count = 1`
7. 开题报告中的 `Paderborn`、蒸馏、Jetson、Atlas、TensorRT 不能写成已完成主结果

## 11. 最终写作顺序

1. 先写第 `5` 章
2. 再写第 `3`、`4`、`6` 章
3. 再写第 `2` 章
4. 再写第 `1` 章
5. 最后写第 `7` 章和中英文摘要

## 12. 一句话原则

整篇论文的所有叙述都要服从这一条：

**以学校格式要求为外层约束，以 `codex/thesis-upgrade` 分支代码和最终实验结果为事实依据，以开题报告为研究动机，不夸大，不越界。**
