# 第5章图表重设计映射

本目录存放论文第5章重设计后的绘图脚本与导出图片，旧版 `Chart/` 目录保留不动。

| 正文图号 | 旧文件 | 新文件 | 数据来源 |
| :-- | --- | --- | --- |
| 图5-1 | `Chart/fig5_2_shot_trend.png`（原图号错位） | `fig5_1_main_experiment_overview.png` | `logs/thesis_tables/controlled/table0_preprocess_model_matrix.csv` |
| 图5-2 | `Chart/fig5_3_main_experiment_heatmap.png` | `fig5_2_main_experiment_matrix_heatmap.png` | `logs/thesis_tables/paper_balanced/table0_preprocess_model_matrix_mean_std.csv` |
| 图5-3 | `Chart/fig5_4_model_stability_errorbar.png` | `fig5_3_model_stability_detail.png` | `table0_preprocess_model_matrix_mean_std.csv` + `seed/seed43/table0_preprocess_model_matrix.csv` + `seed/seed44/table0_preprocess_model_matrix.csv` |
| 图5-4 | `Chart/fig5_5_few_shot_stability.png` | `fig5_4_shot_stability_detail.png` | `table2_few_shot_mean_std.csv` + `seed/seed43/table0_preprocess_model_matrix.csv` + `seed/seed44/table0_preprocess_model_matrix.csv` |
| 图5-5 | `Chart/fig5_6_domain_robustness_and_difficulty.png` | `fig5_5_cross_domain_robustness.png` | `logs/thesis_tables/paper_balanced/table3_domain_robustness.csv` |
| 图5-6 | `Chart/fig5_8_accuracy_vs_model_size.png` | `fig5_6_accuracy_vs_model_size_compression.png` | `logs/thesis_tables/paper_balanced/table4_compression_ablation.csv` |
| 图5-7 | `Chart/fig5_9_accuracy_vs_latency.png` | `fig5_7_accuracy_vs_latency_compression.png` | `logs/thesis_tables/paper_balanced/table4_compression_ablation.csv` |
