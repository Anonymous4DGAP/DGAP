# Dual Graph Adaptive Propagation for Inductive Text Classification

---

## Requirements

python == 3.6

numpy == 1.17

scipy == 1.5.2

nltk == 3.3

stanfordcorenlp == 3.9.1

tqdm == 4.47.0

torch == 1.8.1

torch-cluster == 1.5.9

torch-geometric == 2.0.0

torch-scatter == 2.0.8

torch-sparse == 0.6.9

torch-spline-conv == 1.2.1

---

## File Tree

```
.
├── config.py
├── dataset.py
├── log
│   └── README.md
├── model.py
├── preprocess
│   ├── config.py
│   ├── constructor.py
│   ├── filter.py
│   ├── handle_glove.py
│   ├── __init__.py
│   ├── parser.py
│   └── README.md
├── raw
│   ├── mr.labels.txt
│   ├── mr.texts.txt
│   ├── ohsumed.labels.txt
│   ├── ohsumed.texts.txt
│   ├── R8.labels.txt
│   └── R8.texts.txt
├── source
│   ├── glove.6B.300d.npy
│   ├── glove.6B.300d.txt
│   ├── glove.6B.words.pkl
│   └── README.md
├── temp
│   └── README.md
├── train.py
└── utils.py

```

---

## Usage

Before training the model, please refer to the `README.md` in `\preprocess` to perform data preprocessing and generate the training data.

Example of running DGAP:

`python train.py --dataset mr --graph_mode dual --fuse_mode gate --log_dir log` to train DGAP-G;

`python train.py --dataset R8 --graph_mode dual --fuse_mode atten --log_dir log` to train DGAP-A;

---

## Cite

Please cite our paper if you use this code in your own work:

```
# To be published.
```