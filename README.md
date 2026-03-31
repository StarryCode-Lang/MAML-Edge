# MAML Edge

This repository provides a fault-diagnosis training workflow for CWRU-based few-shot learning with one main model and one comparison model.

The active workflow is:

```text
CWRU
-> 1024 sliding window
-> FFT
-> 5-way 5/10/15-shot tasks
-> FOMAML training
-> ProtoNet comparison
```

## Project Structure

- [train_maml.py](/D:/Desktop/Few-shot-Fault-Diagnosis-MAML/train_maml.py)
  Training entry for the main MAML or FOMAML pipeline.

- [maml.py](/D:/Desktop/Few-shot-Fault-Diagnosis-MAML/maml.py)
  Main training logic for the baseline model.

- [train_protonet.py](/D:/Desktop/Few-shot-Fault-Diagnosis-MAML/train_protonet.py)
  Training entry for the comparison model.

- [protonet.py](/D:/Desktop/Few-shot-Fault-Diagnosis-MAML/protonet.py)
  ProtoNet training logic built on the same data and task pipeline.

- [models.py](/D:/Desktop/Few-shot-Fault-Diagnosis-MAML/models.py)
  Shared model definitions.

- [fault_datasets.py](/D:/Desktop/Few-shot-Fault-Diagnosis-MAML/fault_datasets.py)
  Dataset wrappers for CWRU and HST.

- [preprocess_cwru.py](/D:/Desktop/Few-shot-Fault-Diagnosis-MAML/preprocess_cwru.py)
  CWRU loading and preprocessing.

- [preprocess_hst.py](/D:/Desktop/Few-shot-Fault-Diagnosis-MAML/preprocess_hst.py)
  HST loading and preprocessing.

- [utils.py](/D:/Desktop/Few-shot-Fault-Diagnosis-MAML/utils.py)
  Shared logging, adaptation, accuracy, and data helpers.

- [export_model.py](/D:/Desktop/Few-shot-Fault-Diagnosis-MAML/export_model.py)
  Structured pruning, ONNX export, and INT8 quantization.

- [requirements.txt](/D:/Desktop/Few-shot-Fault-Diagnosis-MAML/requirements.txt)
  Python dependencies.

- [scripts/download_cwru_release.sh](/D:/Desktop/Few-shot-Fault-Diagnosis-MAML/scripts/download_cwru_release.sh)
  Helper script for downloading the CWRU release data.

## Data Layout

Expected dataset layout:

```text
data/
  CWRU_12k/
    Drive_end_0/
    Drive_end_1/
    Drive_end_2/
    Drive_end_3/
```

## Main Settings

The main thesis settings are:

- `dataset = CWRU`
- `preprocess = FFT`
- `window_size = 1024`
- `ways = 5`
- `shots = 5 / 10 / 15`
- `train_domains = 0,1,2`
- `test_domain = 3`
- `meta_train = 10`
- `meta_test = 3`
- repeated runs `>= 5`

## Run Commands

Train the baseline:

```bash
python train_maml.py --ways 5 --shots 5 --iters 300 --first_order True --preprocess FFT --train_domains 0,1,2 --test_domain 3 --meta_batch_size 10 --meta_test_batch_size 3
```

Train the comparison model:

```bash
python train_protonet.py --ways 5 --shots 5 --iters 300 --preprocess FFT --train_domains 0,1,2 --test_domain 3 --meta_batch_size 10 --meta_test_batch_size 3
```

Run the same commands again with `--shots 10` and `--shots 15`.
For thesis reporting, repeat each setting with at least 5 seeds.

## Environment

Recommended environment:

- Python `3.10`
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

Install:

```bash
pip install -r requirements.txt
```

## Notes

- `meta_test_batch_size` is optional in the baseline entry.
- If it is not set, the default behavior stays aligned with the baseline training flow.
- This repository keeps the baseline and the comparison model on the same dataset, preprocessing route, and task construction path.

## References

- [Project reference 1](https://github.com/Yifei20/Few-shot-Fault-Diagnosis-MAML)
- [Project reference 2](https://github.com/fyancy/MetaFD)
