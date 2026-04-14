# Paper Balanced Tables

## Table 0 - Seed Stability Matrix

| preprocess | model | shots | sample_count | metric_protocol | accuracy_mean | accuracy_std | accuracy_mean_percent | accuracy_std_percent | accuracy_mean_std | latency_mean_ms | latency_std_ms | latency_mean_std_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FFT | CNN | 5 | 3 | deployment_baseline_mean_std | 0.687234 | 0.16029 | 68.72 | 16.03 | 68.72 +- 16.03 | 1.1585 | 0.1197 | 1.1585 +- 0.1197 |
| FFT | CNN | 10 | 3 | deployment_baseline_mean_std | 0.740186 | 0.156039 | 74.02 | 15.6 | 74.02 +- 15.6 | 1.1676 | 0.0965 | 1.1676 +- 0.0965 |
| FFT | CNN | 15 | 3 | deployment_baseline_mean_std | 0.668827 | 0.331437 | 66.88 | 33.14 | 66.88 +- 33.14 | 1.0625 | 0.1278 | 1.0625 +- 0.1278 |
| FFT | MAML | 5 | 3 | deployment_baseline_mean_std | 0.897683 | 0.093031 | 89.77 | 9.3 | 89.77 +- 9.3 | 1.1283 | 0.1354 | 1.1283 +- 0.1354 |
| FFT | MAML | 10 | 3 | deployment_baseline_mean_std | 0.998387 | 0.001578 | 99.84 | 0.16 | 99.84 +- 0.16 | 1.2155 | 0.0569 | 1.2155 +- 0.0569 |
| FFT | MAML | 15 | 3 | deployment_baseline_mean_std | 0.886153 | 0.194455 | 88.62 | 19.45 | 88.62 +- 19.45 | 1.1876 | 0.0841 | 1.1876 +- 0.0841 |
| FFT | ProtoNet | 5 | 3 | deployment_baseline_mean_std | 0.999439 | 0.000972 | 99.94 | 0.1 | 99.94 +- 0.1 | 1.0389 | 0.145 | 1.0389 +- 0.145 |
| FFT | ProtoNet | 10 | 3 | deployment_baseline_mean_std | 1.0 | 0.0 | 100.0 | 0.0 | 100.0 +- 0.0 | 1.2233 | 0.0286 | 1.2233 +- 0.0286 |
| FFT | ProtoNet | 15 | 3 | deployment_baseline_mean_std | 1.0 | 0.0 | 100.0 | 0.0 | 100.0 +- 0.0 | 1.2296 | 0.1213 | 1.2296 +- 0.1213 |
| STFT | CNN | 5 | 3 | deployment_baseline_mean_std | 0.821556 | 0.14684 | 82.16 | 14.68 | 82.16 +- 14.68 | 2.738 | 0.202 | 2.738 +- 0.202 |
| STFT | CNN | 10 | 3 | deployment_baseline_mean_std | 0.858729 | 0.12293 | 85.87 | 12.29 | 85.87 +- 12.29 | 2.7218 | 0.1059 | 2.7218 +- 0.1059 |
| STFT | CNN | 15 | 3 | deployment_baseline_mean_std | 0.913538 | 0.066725 | 91.35 | 6.67 | 91.35 +- 6.67 | 2.8424 | 0.2217 | 2.8424 +- 0.2217 |
| STFT | MAML | 5 | 3 | deployment_baseline_mean_std | 0.992145 | 0.007591 | 99.21 | 0.76 | 99.21 +- 0.76 | 2.9231 | 0.0226 | 2.9231 +- 0.0226 |
| STFT | MAML | 10 | 3 | deployment_baseline_mean_std | 0.996424 | 0.003685 | 99.64 | 0.37 | 99.64 +- 0.37 | 2.8125 | 0.1051 | 2.8125 +- 0.1051 |
| STFT | MAML | 15 | 3 | deployment_baseline_mean_std | 0.988413 | 0.015738 | 98.84 | 1.57 | 98.84 +- 1.57 | 2.8466 | 0.1211 | 2.8466 +- 0.1211 |
| STFT | ProtoNet | 5 | 3 | deployment_baseline_mean_std | 0.997194 | 0.00486 | 99.72 | 0.49 | 99.72 +- 0.49 | 2.7307 | 0.0431 | 2.7307 +- 0.0431 |
| STFT | ProtoNet | 10 | 3 | deployment_baseline_mean_std | 0.997755 | 0.003888 | 99.78 | 0.39 | 99.78 +- 0.39 | 2.7885 | 0.1455 | 2.7885 +- 0.1455 |
| STFT | ProtoNet | 15 | 3 | deployment_baseline_mean_std | 0.997194 | 0.00486 | 99.72 | 0.49 | 99.72 +- 0.49 | 2.7195 | 0.0255 | 2.7195 +- 0.0255 |
| WT | CNN | 5 | 3 | deployment_baseline_mean_std | 0.974596 | 0.022856 | 97.46 | 2.29 | 97.46 +- 2.29 | 2.6526 | 0.1201 | 2.6526 +- 0.1201 |
| WT | CNN | 10 | 3 | deployment_baseline_mean_std | 0.953274 | 0.065056 | 95.33 | 6.51 | 95.33 +- 6.51 | 2.6741 | 0.0475 | 2.6741 +- 0.0475 |
| WT | CNN | 15 | 3 | deployment_baseline_mean_std | 0.973691 | 0.035725 | 97.37 | 3.57 | 97.37 +- 3.57 | 2.7586 | 0.1535 | 2.7586 +- 0.1535 |
| WT | MAML | 5 | 3 | deployment_baseline_mean_std | 0.995302 | 0.004081 | 99.53 | 0.41 | 99.53 +- 0.41 | 2.5643 | 0.1569 | 2.5643 +- 0.1569 |
| WT | MAML | 10 | 3 | deployment_baseline_mean_std | 0.994526 | 0.00345 | 99.45 | 0.34 | 99.45 +- 0.34 | 2.686 | 0.237 | 2.686 +- 0.237 |
| WT | MAML | 15 | 3 | deployment_baseline_mean_std | 0.999088 | 0.000851 | 99.91 | 0.09 | 99.91 +- 0.09 | 2.6703 | 0.0392 | 2.6703 +- 0.0392 |
| WT | ProtoNet | 5 | 3 | deployment_baseline_mean_std | 0.997825 | 0.001158 | 99.78 | 0.12 | 99.78 +- 0.12 | 2.5899 | 0.1393 | 2.5899 +- 0.1393 |
| WT | ProtoNet | 10 | 3 | deployment_baseline_mean_std | 0.999088 | 0.000851 | 99.91 | 0.09 | 99.91 +- 0.09 | 2.6727 | 0.0457 | 2.6727 +- 0.0457 |
| WT | ProtoNet | 15 | 3 | deployment_baseline_mean_std | 0.998737 | 0.000364 | 99.87 | 0.04 | 99.87 +- 0.04 | 2.5885 | 0.1506 | 2.5885 +- 0.1506 |

## Table 1 - Model Performance Mean +- Std

| model | sample_count | metric_protocol | accuracy_mean_percent | accuracy_std_percent | accuracy_mean_std |
| --- | --- | --- | --- | --- | --- |
| CNN | 3 | deployment_baseline_mean_std | 82.16 | 14.68 | 82.16 +- 14.68 |
| MAML | 3 | deployment_baseline_mean_std | 99.21 | 0.76 | 99.21 +- 0.76 |
| ProtoNet | 3 | deployment_baseline_mean_std | 99.72 | 0.49 | 99.72 +- 0.49 |

## Table 2 - Few-Shot Mean +- Std

| model | shots | sample_count | metric_protocol | accuracy_mean_percent | accuracy_std_percent | accuracy_mean_std |
| --- | --- | --- | --- | --- | --- | --- |
| MAML | 5 | 3 | deployment_baseline_mean_std | 99.21 | 0.76 | 99.21 +- 0.76 |
| MAML | 10 | 3 | deployment_baseline_mean_std | 99.64 | 0.37 | 99.64 +- 0.37 |
| MAML | 15 | 3 | deployment_baseline_mean_std | 98.84 | 1.57 | 98.84 +- 1.57 |

## Table 3 - Domain Robustness

| train_domains | test_domain | preprocess | model | shots | seed | metric_protocol | accuracy | accuracy_percent | latency_ms | experiment_title |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0,1,2 | 3 | FFT | CNN | 5 | 42 | deployment_baseline | 0.85068349106204 | 85.07 | 1.2193 | CNN_CWRU_FFT_5w5s5q_source012_target3_labels0123456789 |
| 0,1,2 | 3 | FFT | CNN | 10 | 42 | deployment_baseline | 0.9148264984227129 | 91.48 | 1.0674 | CNN_CWRU_FFT_5w10s10q_source012_target3_labels0123456789 |
| 0,1,2 | 3 | FFT | CNN | 15 | 42 | deployment_baseline | 0.9190325972660357 | 91.9 | 1.1826 | CNN_CWRU_FFT_5w15s15q_source012_target3_labels0123456789 |
| 0,1,2 | 3 | FFT | MAML | 5 | 42 | deployment_baseline | 0.8748685594111462 | 87.49 | 1.2277 | MAML_CWRU_FFT_5w5s_source012_target3_labels0123456789 |
| 0,1,2 | 3 | FFT | MAML | 10 | 42 | deployment_baseline | 0.9968454258675079 | 99.68 | 1.2684 | MAML_CWRU_FFT_5w10s_source012_target3_labels0123456789 |
| 0,1,2 | 3 | FFT | MAML | 15 | 42 | deployment_baseline | 0.9978969505783386 | 99.79 | 1.2062 | MAML_CWRU_FFT_5w15s_source012_target3_labels0123456789 |
| 0,1,2 | 3 | FFT | ProtoNet | 5 | 42 | deployment_baseline | 1.0 | 100.0 | 1.2038 | ProtoNet_CWRU_FFT_5w5s5q_source012_target3_labels0123456789 |
| 0,1,2 | 3 | FFT | ProtoNet | 10 | 42 | deployment_baseline | 1.0 | 100.0 | 1.2217 | ProtoNet_CWRU_FFT_5w10s10q_source012_target3_labels0123456789 |
| 0,1,2 | 3 | FFT | ProtoNet | 15 | 42 | deployment_baseline | 1.0 | 100.0 | 1.2075 | ProtoNet_CWRU_FFT_5w15s15q_source012_target3_labels0123456789 |
| 0,1,2 | 3 | STFT | CNN | 5 | 42 | deployment_baseline | 0.9905362776025236 | 99.05 | 2.7751 | CNN_CWRU_STFT_5w5s5q_source012_target3_labels0123456789 |
| 0,1,2 | 3 | STFT | CNN | 10 | 42 | deployment_baseline | 0.9831756046267087 | 98.32 | 2.7819 | CNN_CWRU_STFT_5w10s10q_source012_target3_labels0123456789 |
| 0,1,2 | 3 | STFT | CNN | 15 | 42 | deployment_baseline | 0.9831756046267087 | 98.32 | 2.7181 | CNN_CWRU_STFT_5w15s15q_source012_target3_labels0123456789 |
| 0,1,2 | 3 | STFT | MAML | 5 | 42 | deployment_baseline | 0.9915878023133544 | 99.16 | 2.899 | MAML_CWRU_STFT_5w5s_source012_target3_labels0123456789 |
| 0,1,2 | 3 | STFT | MAML | 10 | 42 | deployment_baseline | 0.9926393270241851 | 99.26 | 2.7164 | MAML_CWRU_STFT_5w10s_source012_target3_labels0123456789 |
| 0,1,2 | 3 | STFT | MAML | 15 | 42 | deployment_baseline | 0.9947423764458465 | 99.47 | 2.7156 | MAML_CWRU_STFT_5w15s_source012_target3_labels0123456789 |
| 0,1,2 | 3 | STFT | ProtoNet | 5 | 42 | deployment_baseline | 1.0 | 100.0 | 2.7264 | ProtoNet_CWRU_STFT_5w5s5q_source012_target3_labels0123456789 |
| 0,1,2 | 3 | STFT | ProtoNet | 10 | 42 | deployment_baseline | 1.0 | 100.0 | 2.7676 | ProtoNet_CWRU_STFT_5w10s10q_source012_target3_labels0123456789 |
| 0,1,2 | 3 | STFT | ProtoNet | 15 | 42 | deployment_baseline | 1.0 | 100.0 | 2.729 | ProtoNet_CWRU_STFT_5w15s15q_source012_target3_labels0123456789 |
| 0,1,2 | 3 | WT | CNN | 5 | 42 | deployment_baseline | 0.9968454258675079 | 99.68 | 2.7852 | CNN_CWRU_WT_5w5s5q_source012_target3_labels0123456789 |
| 0,1,2 | 3 | WT | CNN | 10 | 42 | deployment_baseline | 0.9989484752891693 | 99.89 | 2.6192 | CNN_CWRU_WT_5w10s10q_source012_target3_labels0123456789 |
| 0,1,2 | 3 | WT | CNN | 15 | 42 | deployment_baseline | 0.9978969505783386 | 99.79 | 2.6741 | CNN_CWRU_WT_5w15s15q_source012_target3_labels0123456789 |
| 0,1,2 | 3 | WT | MAML | 5 | 42 | deployment_baseline | 0.9926393270241851 | 99.26 | 2.5968 | MAML_CWRU_WT_5w5s_source012_target3_labels0123456789 |
| 0,1,2 | 3 | WT | MAML | 10 | 42 | deployment_baseline | 0.9936908517350158 | 99.37 | 2.7706 | MAML_CWRU_WT_5w10s_source012_target3_labels0123456789 |
| 0,1,2 | 3 | WT | MAML | 15 | 42 | deployment_baseline | 1.0 | 100.0 | 2.673 | MAML_CWRU_WT_5w15s_source012_target3_labels0123456789 |
| 0,1,2 | 3 | WT | ProtoNet | 5 | 42 | deployment_baseline | 0.9978969505783386 | 99.79 | 2.6308 | ProtoNet_CWRU_WT_5w5s5q_source012_target3_labels0123456789 |
| 0,1,2 | 3 | WT | ProtoNet | 10 | 42 | deployment_baseline | 1.0 | 100.0 | 2.6244 | ProtoNet_CWRU_WT_5w10s10q_source012_target3_labels0123456789 |
| 0,1,2 | 3 | WT | ProtoNet | 15 | 42 | deployment_baseline | 0.9989484752891693 | 99.89 | 2.4148 | ProtoNet_CWRU_WT_5w15s15q_source012_target3_labels0123456789 |
| 0,1,3 | 2 | FFT | CNN | 5 | 42 | deployment_baseline | 0.8786919831223629 | 87.87 | 1.2623 | CNN_CWRU_FFT_5w5s5q_source013_target2_labels0123456789 |
| 0,1,3 | 2 | FFT | CNN | 10 | 42 | deployment_baseline | 0.8744725738396625 | 87.45 | 1.269 | CNN_CWRU_FFT_5w10s10q_source013_target2_labels0123456789 |
| 0,1,3 | 2 | FFT | CNN | 15 | 42 | deployment_baseline | 0.890295358649789 | 89.03 | 1.2205 | CNN_CWRU_FFT_5w15s15q_source013_target2_labels0123456789 |
| 0,1,3 | 2 | FFT | MAML | 5 | 42 | deployment_baseline | 0.9957805907172996 | 99.58 | 1.1955 | MAML_CWRU_FFT_5w5s_source013_target2_labels0123456789 |
| 0,1,3 | 2 | FFT | MAML | 10 | 42 | deployment_baseline | 0.9926160337552743 | 99.26 | 1.2291 | MAML_CWRU_FFT_5w10s_source013_target2_labels0123456789 |
| 0,1,3 | 2 | FFT | MAML | 15 | 42 | deployment_baseline | 1.0 | 100.0 | 1.3629 | MAML_CWRU_FFT_5w15s_source013_target2_labels0123456789 |
| 0,1,3 | 2 | FFT | ProtoNet | 5 | 42 | deployment_baseline | 1.0 | 100.0 | 1.1701 | ProtoNet_CWRU_FFT_5w5s5q_source013_target2_labels0123456789 |
| 0,1,3 | 2 | FFT | ProtoNet | 10 | 42 | deployment_baseline | 1.0 | 100.0 | 1.097 | ProtoNet_CWRU_FFT_5w10s10q_source013_target2_labels0123456789 |
| 0,1,3 | 2 | FFT | ProtoNet | 15 | 42 | deployment_baseline | 1.0 | 100.0 | 1.2365 | ProtoNet_CWRU_FFT_5w15s15q_source013_target2_labels0123456789 |
| 0,1,3 | 2 | STFT | CNN | 5 | 42 | deployment_baseline | 0.9746835443037974 | 97.47 | 2.7307 | CNN_CWRU_STFT_5w5s5q_source013_target2_labels0123456789 |
| 0,1,3 | 2 | STFT | CNN | 10 | 42 | deployment_baseline | 0.9462025316455697 | 94.62 | 3.0077 | CNN_CWRU_STFT_5w10s10q_source013_target2_labels0123456789 |
| 0,1,3 | 2 | STFT | CNN | 15 | 42 | deployment_baseline | 0.9746835443037974 | 97.47 | 2.5726 | CNN_CWRU_STFT_5w15s15q_source013_target2_labels0123456789 |
| 0,1,3 | 2 | STFT | MAML | 5 | 42 | deployment_baseline | 0.9947257383966245 | 99.47 | 2.6803 | MAML_CWRU_STFT_5w5s_source013_target2_labels0123456789 |
| 0,1,3 | 2 | STFT | MAML | 10 | 42 | deployment_baseline | 0.9989451476793249 | 99.89 | 2.3696 | MAML_CWRU_STFT_5w10s_source013_target2_labels0123456789 |
| 0,1,3 | 2 | STFT | MAML | 15 | 42 | deployment_baseline | 1.0 | 100.0 | 2.6757 | MAML_CWRU_STFT_5w15s_source013_target2_labels0123456789 |
| 0,1,3 | 2 | STFT | ProtoNet | 5 | 42 | deployment_baseline | 1.0 | 100.0 | 2.686 | ProtoNet_CWRU_STFT_5w5s5q_source013_target2_labels0123456789 |
| 0,1,3 | 2 | STFT | ProtoNet | 10 | 42 | deployment_baseline | 1.0 | 100.0 | 2.5214 | ProtoNet_CWRU_STFT_5w10s10q_source013_target2_labels0123456789 |
| 0,1,3 | 2 | STFT | ProtoNet | 15 | 42 | deployment_baseline | 1.0 | 100.0 | 2.7864 | ProtoNet_CWRU_STFT_5w15s15q_source013_target2_labels0123456789 |
| 0,1,3 | 2 | WT | CNN | 5 | 42 | deployment_baseline | 0.9483122362869199 | 94.83 | 2.6458 | CNN_CWRU_WT_5w5s5q_source013_target2_labels0123456789 |
| 0,1,3 | 2 | WT | CNN | 10 | 42 | deployment_baseline | 0.9409282700421941 | 94.09 | 2.479 | CNN_CWRU_WT_5w10s10q_source013_target2_labels0123456789 |
| 0,1,3 | 2 | WT | CNN | 15 | 42 | deployment_baseline | 0.9483122362869199 | 94.83 | 2.4696 | CNN_CWRU_WT_5w15s15q_source013_target2_labels0123456789 |
| 0,1,3 | 2 | WT | MAML | 5 | 42 | deployment_baseline | 0.9978902953586498 | 99.79 | 2.7593 | MAML_CWRU_WT_5w5s_source013_target2_labels0123456789 |
| 0,1,3 | 2 | WT | MAML | 10 | 42 | deployment_baseline | 0.9926160337552743 | 99.26 | 2.6715 | MAML_CWRU_WT_5w10s_source013_target2_labels0123456789 |
| 0,1,3 | 2 | WT | MAML | 15 | 42 | deployment_baseline | 0.9989451476793249 | 99.89 | 2.5744 | MAML_CWRU_WT_5w15s_source013_target2_labels0123456789 |
| 0,1,3 | 2 | WT | ProtoNet | 5 | 42 | deployment_baseline | 1.0 | 100.0 | 2.5578 | ProtoNet_CWRU_WT_5w5s5q_source013_target2_labels0123456789 |
| 0,1,3 | 2 | WT | ProtoNet | 10 | 42 | deployment_baseline | 1.0 | 100.0 | 2.6533 | ProtoNet_CWRU_WT_5w10s10q_source013_target2_labels0123456789 |
| 0,1,3 | 2 | WT | ProtoNet | 15 | 42 | deployment_baseline | 1.0 | 100.0 | 2.7214 | ProtoNet_CWRU_WT_5w15s15q_source013_target2_labels0123456789 |
| 0,2,3 | 1 | FFT | CNN | 5 | 42 | deployment_baseline | 0.8974630021141649 | 89.75 | 1.4874 | CNN_CWRU_FFT_5w5s5q_source023_target1_labels0123456789 |
| 0,2,3 | 1 | FFT | CNN | 10 | 42 | deployment_baseline | 0.8773784355179705 | 87.74 | 1.0958 | CNN_CWRU_FFT_5w10s10q_source023_target1_labels0123456789 |
| 0,2,3 | 1 | FFT | CNN | 15 | 42 | deployment_baseline | 0.879492600422833 | 87.95 | 1.3935 | CNN_CWRU_FFT_5w15s15q_source023_target1_labels0123456789 |
| 0,2,3 | 1 | FFT | MAML | 5 | 42 | deployment_baseline | 0.9936575052854123 | 99.37 | 1.4332 | MAML_CWRU_FFT_5w5s_source023_target1_labels0123456789 |
| 0,2,3 | 1 | FFT | MAML | 10 | 42 | deployment_baseline | 0.9936575052854123 | 99.37 | 1.3931 | MAML_CWRU_FFT_5w10s_source023_target1_labels0123456789 |
| 0,2,3 | 1 | FFT | MAML | 15 | 42 | deployment_baseline | 0.9915433403805497 | 99.15 | 1.4848 | MAML_CWRU_FFT_5w15s_source023_target1_labels0123456789 |
| 0,2,3 | 1 | FFT | ProtoNet | 5 | 42 | deployment_baseline | 1.0 | 100.0 | 1.3136 | ProtoNet_CWRU_FFT_5w5s5q_source023_target1_labels0123456789 |
| 0,2,3 | 1 | FFT | ProtoNet | 10 | 42 | deployment_baseline | 0.9936575052854123 | 99.37 | 1.3473 | ProtoNet_CWRU_FFT_5w10s10q_source023_target1_labels0123456789 |
| 0,2,3 | 1 | FFT | ProtoNet | 15 | 42 | deployment_baseline | 1.0 | 100.0 | 1.1786 | ProtoNet_CWRU_FFT_5w15s15q_source023_target1_labels0123456789 |
| 0,2,3 | 1 | STFT | CNN | 5 | 42 | deployment_baseline | 0.9608879492600423 | 96.09 | 2.6819 | CNN_CWRU_STFT_5w5s5q_source023_target1_labels0123456789 |
| 0,2,3 | 1 | STFT | CNN | 10 | 42 | deployment_baseline | 0.9534883720930233 | 95.35 | 2.7413 | CNN_CWRU_STFT_5w10s10q_source023_target1_labels0123456789 |
| 0,2,3 | 1 | STFT | CNN | 15 | 42 | deployment_baseline | 0.9841437632135307 | 98.41 | 2.2351 | CNN_CWRU_STFT_5w15s15q_source023_target1_labels0123456789 |
| 0,2,3 | 1 | STFT | MAML | 5 | 42 | deployment_baseline | 0.9915433403805497 | 99.15 | 2.8632 | MAML_CWRU_STFT_5w5s_source023_target1_labels0123456789 |
| 0,2,3 | 1 | STFT | MAML | 10 | 42 | deployment_baseline | 0.9957716701902748 | 99.58 | 2.7204 | MAML_CWRU_STFT_5w10s_source023_target1_labels0123456789 |
| 0,2,3 | 1 | STFT | MAML | 15 | 42 | deployment_baseline | 0.9978858350951374 | 99.79 | 2.9149 | MAML_CWRU_STFT_5w15s_source023_target1_labels0123456789 |
| 0,2,3 | 1 | STFT | ProtoNet | 5 | 42 | deployment_baseline | 0.9978858350951374 | 99.79 | 2.8486 | ProtoNet_CWRU_STFT_5w5s5q_source023_target1_labels0123456789 |
| 0,2,3 | 1 | STFT | ProtoNet | 10 | 42 | deployment_baseline | 0.9968287526427061 | 99.68 | 2.8373 | ProtoNet_CWRU_STFT_5w10s10q_source023_target1_labels0123456789 |
| 0,2,3 | 1 | STFT | ProtoNet | 15 | 42 | deployment_baseline | 1.0 | 100.0 | 2.7441 | ProtoNet_CWRU_STFT_5w15s15q_source023_target1_labels0123456789 |
| 0,2,3 | 1 | WT | CNN | 5 | 42 | deployment_baseline | 0.9154334038054969 | 91.54 | 2.7393 | CNN_CWRU_WT_5w5s5q_source023_target1_labels0123456789 |
| 0,2,3 | 1 | WT | CNN | 10 | 42 | deployment_baseline | 0.9799154334038055 | 97.99 | 2.628 | CNN_CWRU_WT_5w10s10q_source023_target1_labels0123456789 |
| 0,2,3 | 1 | WT | CNN | 15 | 42 | deployment_baseline | 0.96723044397463 | 96.72 | 2.5978 | CNN_CWRU_WT_5w15s15q_source023_target1_labels0123456789 |
| 0,2,3 | 1 | WT | MAML | 5 | 42 | deployment_baseline | 0.992600422832981 | 99.26 | 2.6828 | MAML_CWRU_WT_5w5s_source023_target1_labels0123456789 |
| 0,2,3 | 1 | WT | MAML | 10 | 42 | deployment_baseline | 0.9957716701902748 | 99.58 | 2.7169 | MAML_CWRU_WT_5w10s_source023_target1_labels0123456789 |
| 0,2,3 | 1 | WT | MAML | 15 | 42 | deployment_baseline | 0.9862579281183932 | 98.63 | 2.9436 | MAML_CWRU_WT_5w15s_source023_target1_labels0123456789 |
| 0,2,3 | 1 | WT | ProtoNet | 5 | 42 | deployment_baseline | 0.9883720930232558 | 98.84 | 2.7634 | ProtoNet_CWRU_WT_5w5s5q_source023_target1_labels0123456789 |
| 0,2,3 | 1 | WT | ProtoNet | 10 | 42 | deployment_baseline | 0.9936575052854123 | 99.37 | 2.7309 | ProtoNet_CWRU_WT_5w10s10q_source023_target1_labels0123456789 |
| 0,2,3 | 1 | WT | ProtoNet | 15 | 42 | deployment_baseline | 0.9904862579281184 | 99.05 | 2.562 | ProtoNet_CWRU_WT_5w15s15q_source023_target1_labels0123456789 |
| 1,2,3 | 0 | FFT | CNN | 5 | 42 | deployment_baseline | 0.9634831460674157 | 96.35 | 0.9705 | CNN_CWRU_FFT_5w5s5q_source123_target0_labels0123456789 |
| 1,2,3 | 0 | FFT | CNN | 10 | 42 | deployment_baseline | 0.875 | 87.5 | 1.2666 | CNN_CWRU_FFT_5w10s10q_source123_target0_labels0123456789 |
| 1,2,3 | 0 | FFT | CNN | 15 | 42 | deployment_baseline | 0.9283707865168539 | 92.84 | 1.1872 | CNN_CWRU_FFT_5w15s15q_source123_target0_labels0123456789 |
| 1,2,3 | 0 | FFT | MAML | 5 | 42 | deployment_baseline | 0.9747191011235955 | 97.47 | 1.2303 | MAML_CWRU_FFT_5w5s_source123_target0_labels0123456789 |
| 1,2,3 | 0 | FFT | MAML | 10 | 42 | deployment_baseline | 0.9452247191011236 | 94.52 | 1.1513 | MAML_CWRU_FFT_5w10s_source123_target0_labels0123456789 |
| 1,2,3 | 0 | FFT | MAML | 15 | 42 | deployment_baseline | 1.0 | 100.0 | 1.1018 | MAML_CWRU_FFT_5w15s_source123_target0_labels0123456789 |
| 1,2,3 | 0 | FFT | ProtoNet | 5 | 42 | deployment_baseline | 0.9985955056179775 | 99.86 | 1.2587 | ProtoNet_CWRU_FFT_5w5s5q_source123_target0_labels0123456789 |
| 1,2,3 | 0 | FFT | ProtoNet | 10 | 42 | deployment_baseline | 0.9985955056179775 | 99.86 | 1.1942 | ProtoNet_CWRU_FFT_5w10s10q_source123_target0_labels0123456789 |
| 1,2,3 | 0 | FFT | ProtoNet | 15 | 42 | deployment_baseline | 1.0 | 100.0 | 1.2404 | ProtoNet_CWRU_FFT_5w15s15q_source123_target0_labels0123456789 |
| 1,2,3 | 0 | STFT | CNN | 5 | 42 | deployment_baseline | 0.8160112359550562 | 81.6 | 2.718 | CNN_CWRU_STFT_5w5s5q_source123_target0_labels0123456789 |
| 1,2,3 | 0 | STFT | CNN | 10 | 42 | deployment_baseline | 0.9367977528089888 | 93.68 | 2.8804 | CNN_CWRU_STFT_5w10s10q_source123_target0_labels0123456789 |
| 1,2,3 | 0 | STFT | CNN | 15 | 42 | deployment_baseline | 0.9353932584269663 | 93.54 | 2.7157 | CNN_CWRU_STFT_5w15s15q_source123_target0_labels0123456789 |
| 1,2,3 | 0 | STFT | MAML | 5 | 42 | deployment_baseline | 0.9676966292134831 | 96.77 | 2.889 | MAML_CWRU_STFT_5w5s_source123_target0_labels0123456789 |
| 1,2,3 | 0 | STFT | MAML | 10 | 42 | deployment_baseline | 0.973314606741573 | 97.33 | 2.6667 | MAML_CWRU_STFT_5w10s_source123_target0_labels0123456789 |
| 1,2,3 | 0 | STFT | MAML | 15 | 42 | deployment_baseline | 0.9634831460674157 | 96.35 | 2.8577 | MAML_CWRU_STFT_5w15s_source123_target0_labels0123456789 |
| 1,2,3 | 0 | STFT | ProtoNet | 5 | 42 | deployment_baseline | 0.9817415730337079 | 98.17 | 2.8 | ProtoNet_CWRU_STFT_5w5s5q_source123_target0_labels0123456789 |
| 1,2,3 | 0 | STFT | ProtoNet | 10 | 42 | deployment_baseline | 0.973314606741573 | 97.33 | 2.7818 | ProtoNet_CWRU_STFT_5w10s10q_source123_target0_labels0123456789 |
| 1,2,3 | 0 | STFT | ProtoNet | 15 | 42 | deployment_baseline | 0.9873595505617978 | 98.74 | 2.6528 | ProtoNet_CWRU_STFT_5w15s15q_source123_target0_labels0123456789 |
| 1,2,3 | 0 | WT | CNN | 5 | 42 | deployment_baseline | 0.8609550561797753 | 86.1 | 2.7304 | CNN_CWRU_WT_5w5s5q_source123_target0_labels0123456789 |
| 1,2,3 | 0 | WT | CNN | 10 | 42 | deployment_baseline | 0.922752808988764 | 92.28 | 2.7657 | CNN_CWRU_WT_5w10s10q_source123_target0_labels0123456789 |
| 1,2,3 | 0 | WT | CNN | 15 | 42 | deployment_baseline | 0.9157303370786517 | 91.57 | 2.745 | CNN_CWRU_WT_5w15s15q_source123_target0_labels0123456789 |
| 1,2,3 | 0 | WT | MAML | 5 | 42 | deployment_baseline | 0.9339887640449438 | 93.4 | 2.873 | MAML_CWRU_WT_5w5s_source123_target0_labels0123456789 |
| 1,2,3 | 0 | WT | MAML | 10 | 42 | deployment_baseline | 0.9789325842696629 | 97.89 | 3.0974 | MAML_CWRU_WT_5w10s_source123_target0_labels0123456789 |
| 1,2,3 | 0 | WT | MAML | 15 | 42 | deployment_baseline | 0.9508426966292135 | 95.08 | 2.7979 | MAML_CWRU_WT_5w15s_source123_target0_labels0123456789 |
| 1,2,3 | 0 | WT | ProtoNet | 5 | 42 | deployment_baseline | 0.9901685393258427 | 99.02 | 2.8875 | ProtoNet_CWRU_WT_5w5s5q_source123_target0_labels0123456789 |
| 1,2,3 | 0 | WT | ProtoNet | 10 | 42 | deployment_baseline | 0.9929775280898876 | 99.3 | 2.6654 | ProtoNet_CWRU_WT_5w10s10q_source123_target0_labels0123456789 |
| 1,2,3 | 0 | WT | ProtoNet | 15 | 42 | deployment_baseline | 0.9943820224719101 | 99.44 | 2.868 | ProtoNet_CWRU_WT_5w15s15q_source123_target0_labels0123456789 |

## Table 4 - Compression Ablation

| profile | variant | metric_protocol | accuracy | accuracy_percent | latency_ms | parameter_count | model_size_mb | runtime_backend |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FFT + ProtoNet + 5-shot | baseline | deployment_baseline | 1.0 | 100.0 | 1.3666 | 19008 | 0.073649 | onnxruntime |
| FFT + ProtoNet + 5-shot | int8_only | deployment_baseline_int8 | 1.0 | 100.0 | 0.8214 | 19008 | 0.0292 | onnxruntime |
| FFT + ProtoNet + 5-shot | prune_only | deployment_prune_only | 1.0 | 100.0 | 0.914 | 6840 | 0.027724 | onnxruntime |
| FFT + ProtoNet + 5-shot | prune_recovery_float | deployment_float | 1.0 | 100.0 | 0.9032 | 6840 | 0.027724 | onnxruntime |
| FFT + ProtoNet + 5-shot | prune_recovery_int8 | deployment_int8 | 1.0 | 100.0 | 0.2813 | 6840 | 0.016719 | onnxruntime |
| STFT + MAML + 5-shot | baseline | deployment_baseline | 0.9905362776025236 | 99.05 | 3.0336 | 113738 | 0.436068 | onnxruntime |
| STFT + MAML + 5-shot | int8_only | deployment_baseline_int8 | 0.9915878023133544 | 99.16 | 2.7794 | 113738 | 0.127833 | onnxruntime |
| STFT + MAML + 5-shot | prune_only | deployment_prune_only | 0.982124079915878 | 98.21 | 2.9647 | 40860 | 0.158854 | onnxruntime |
| STFT + MAML + 5-shot | prune_recovery_float | deployment_float | 0.9884332281808622 | 98.84 | 2.9631 | 40860 | 0.158854 | onnxruntime |
| STFT + MAML + 5-shot | prune_recovery_int8 | deployment_int8 | 0.9884332281808622 | 98.84 | 3.3336 | 40860 | 0.056937 | onnxruntime |

## Table 5 - System Performance

| channel | stage | request_count | latency_ms |
| --- | --- | --- | --- |
| direct | preprocess | 100 | 56.2379 |
| direct | inference | 100 | 0.9052 |
| direct | end_to_end | 100 | 57.2581 |
