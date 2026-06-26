# MSBR-KT

This repository provides the implementation of **MSBR-KT**, a multi-source structured-bias routing framework for knowledge tracing.

## Overview

MSBR-KT is designed for student performance prediction in knowledge tracing. The model incorporates multiple structured bias sources, including temporal bias, question-level semantic bias, skill-level semantic bias, direct transition path bias, correct-transition path bias, and wrong-transition path bias.

## Project Structure

```text
MSBR-KT/
├── msbr_kt/
│   ├── __init__.py
│   ├── model.py
│   ├── bias.py
│   ├── layers.py
│   ├── relations.py
│   ├── data.py
│   ├── training.py
│   ├── metrics.py
│   ├── config.py
│   └── utils.py
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

The processed data is not stored directly in the main repository due to file size limitations. Please download it from the release attachment.

## Requirements

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

The main dependencies are:

```text
torch
numpy
scikit-learn
```

## Processed Data

The processed datasets required for training and evaluation are provided as a release attachment.

Please download `data_process.zip` from the latest release:

```text
https://github.com/JiangYu828/MSBR-KT/releases/latest/download/data_process.zip
```

After downloading, unzip it into the root directory of this repository. The expected structure is:

```text
MSBR-KT/
├── data_process/
│   ├── data09/
│   │   ├── train_raw.pkl
│   │   ├── valid_raw.pkl
│   │   ├── test_raw.pkl
│   │   ├── global_maps.pt
│   │   └── relations.pt
│   ├── assist2017/
│   │   ├── train_raw.pkl
│   │   ├── valid_raw.pkl
│   │   ├── test_raw.pkl
│   │   ├── global_maps.pt
│   │   └── relations.pt
│   └── junyi/
│       ├── train_raw.pkl
│       ├── valid_raw.pkl
│       ├── test_raw.pkl
│       ├── global_maps.pt
│       └── relations.pt
├── msbr_kt/
├── train.py
└── evaluate.py
```


## Training

Train MSBR-KT on ASSIST2009:

```bash
python train.py --data_root data_process/data09 --run_dir runs/data09
```

Train MSBR-KT on ASSIST2017:

```bash
python train.py --data_root data_process/assist2017 --run_dir runs/assist2017
```

Train MSBR-KT on Junyi:

```bash
python train.py --data_root data_process/junyi --run_dir runs/junyi
```

You can also specify common hyperparameters manually:

```bash
python train.py --data_root data_process/data09 --run_dir runs/data09 --seq_len 100 --d_model 128 --n_heads 4 --n_layers 1 --batch_size 64 --lr 0.001 --max_epoch 100 --patience 10
```

## Evaluation

After training, evaluate the saved checkpoint with:

```bash
python evaluate.py --data_root data_process/data09 --run_dir runs/data09
```

If you want to specify a checkpoint path manually, use:

```bash
python evaluate.py --data_root data_process/data09 --run_dir runs/data09 --checkpoint runs/data09/best.ckpt
```

You can also dump prediction results with routing weights:

```bash
python evaluate.py --data_root data_process/data09 --run_dir runs/data09 --dump_pred_csv runs/data09/predictions.csv
```

## Notes

Each processed dataset folder should contain the following five files:

```text
train_raw.pkl
valid_raw.pkl
test_raw.pkl
global_maps.pt
relations.pt
```


## Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{msbrkt,
  title={MSBR-KT: Multi-Source Structured-Bias Routing for Knowledge Tracing},
  author={},
  journal={},
  year={}
}
```
