# MAML Application on Few-shot Fault Diagnosis (PyTorch)

This repository now follows one strict rule:

- keep the original `Yifei20/Few-shot-Fault-Diagnosis-MAML` MAML pipeline as the baseline
- only add one comparison line: `ProtoNet`
- do not rebuild the project into a new framework

So the repository should be understood as:

1. original `Yifei20` MAML or FOMAML baseline
2. one extra `ProtoNet` comparison path under the same data and task construction

## 1. Baseline Rule

The main line remains the original Yifei20 logic:

- same CWRU data source
- same sliding window length: `1024`
- same FFT preprocessing route
- same source-domain and target-domain task construction style
- same `learn2learn` training stack
- same environment requirements

Only one small extension is added to the original MAML path:

- `meta_test_batch_size`

This argument is optional.
If you do not set it, the code behaves like the original Yifei20 implementation and uses `meta_batch_size` for both meta-train and meta-test.
If you set it, you can run your thesis setting such as `meta-train = 10` and `meta-test = 3`.

## 2. Thesis Route

For your thesis, the active route should be:

```text
CWRU
-> sliding window 1024
-> FFT
-> 5-way 5/10/15-shot
-> FOMAML as the main model
-> ProtoNet as the comparison model
```

If there is any conflict between this route and the original Yifei20 implementation, this repository keeps the Yifei20 logic first and only adds the comparison model with minimal changes.

## 3. File Roles

- [train_maml.py](/D:/Desktop/diploma/Few-shot-Fault-Diagnosis-MAML/train_maml.py)
  Original MAML training entry.

- [maml.py](/D:/Desktop/diploma/Few-shot-Fault-Diagnosis-MAML/maml.py)
  Original Yifei20 MAML training logic, with only one optional `meta_test_batch_size` extension.

- [models.py](/D:/Desktop/diploma/Few-shot-Fault-Diagnosis-MAML/models.py)
  Contains the original `CNN1D` and the added `ProtoNet1D`.
  The added `ProtoNet1D` reuses the same backbone structure as the Yifei20 `CNN1D` feature extractor to keep the comparison closer.

- [utils.py](/D:/Desktop/diploma/Few-shot-Fault-Diagnosis-MAML/utils.py)
  Keeps the original Yifei20 utility functions and only adds `fast_adapt_protonet`.

- [fault_datasets.py](/D:/Desktop/diploma/Few-shot-Fault-Diagnosis-MAML/fault_datasets.py)
  Original dataset wrapper logic.

- [preprocess_cwru.py](/D:/Desktop/diploma/Few-shot-Fault-Diagnosis-MAML/preprocess_cwru.py)
  Original CWRU preprocessing script.

- [preprocess_hst.py](/D:/Desktop/diploma/Few-shot-Fault-Diagnosis-MAML/preprocess_hst.py)
  Original HST preprocessing script.

- [train_protonet.py](/D:/Desktop/diploma/Few-shot-Fault-Diagnosis-MAML/train_protonet.py)
  New comparison entry. This is the only new training entry added in this rebuild.

- [protonet.py](/D:/Desktop/diploma/Few-shot-Fault-Diagnosis-MAML/protonet.py)
  New comparison training logic.
  It reuses the same CWRU FFT dataset path, the same domain split style, and the same `learn2learn` task construction style as the Yifei20 MAML baseline.

## 4. Environment

The environment is kept consistent with the original Yifei20 project:

- `h5py==3.2.1`
- `learn2learn==0.2.0`
- `matplotlib==3.4.3`
- `numpy==1.22.0`
- `Pillow==10.3.0`
- `PyWavelets==1.1.1`
- `scikit_learn==0.24.2`
- `scipy==1.7.1`
- `torch==2.1.1`
- `torchvision==0.16.1`

Install with:

```shell
pip install -r requirements.txt
```

## 5. Data Preparation

The CWRU raw dataset layout remains:

```text
data/
  CWRU_12k/
    Drive_end_0/
    Drive_end_1/
    Drive_end_2/
    Drive_end_3/
```

For the current thesis route, use the raw dataset with `FFT`.
You do not need the pre-generated `WT` or `STFT` images unless you want to reproduce those branches from the original repository.

## 6. Running the Original MAML Baseline

To stay as close as possible to Yifei20, use the original entry:

```shell
python train_maml.py --ways 5 --shots 5 --iters 300 --first_order True --preprocess FFT --train_domains 0,1,2 --test_domain 3
```

For your thesis setting with separate meta-train and meta-test task counts:

```shell
python train_maml.py --ways 5 --shots 5 --iters 300 --first_order True --preprocess FFT --train_domains 0,1,2 --test_domain 3 --meta_batch_size 10 --meta_test_batch_size 3
```

Run the same command again with:

- `--shots 10`
- `--shots 15`

Repeat each setting with at least 5 different seeds if you want the thesis statistics.

## 7. Running the ProtoNet Comparison

ProtoNet is intentionally restricted to the same active thesis path:

- `dataset = CWRU`
- `preprocess = FFT`

Example:

```shell
python train_protonet.py --ways 5 --shots 5 --iters 300 --preprocess FFT --train_domains 0,1,2 --test_domain 3 --meta_batch_size 10 --meta_test_batch_size 3
```

Run the same command again with:

- `--shots 10`
- `--shots 15`

## 8. Control Variables

The comparison between `FOMAML` and `ProtoNet` is controlled through these shared settings:

- same environment
- same CWRU dataset
- same FFT preprocessing route
- same source domains
- same target domain
- same `N-way K-shot` task construction style
- same plotting, logging, and checkpoint layout

The method itself is the only intended variable.

## 9. Notes

- This repository is no longer treated as a new re-architecture.
- It is the original Yifei20 baseline plus one controlled comparison line.
- If you want an even stricter baseline check, run `train_maml.py` first and ignore `ProtoNet` until the MAML curve matches your expectation.
