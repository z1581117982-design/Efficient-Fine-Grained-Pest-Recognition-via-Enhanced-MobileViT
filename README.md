[README.md](https://github.com/user-attachments/files/34546454/README.md)
# Efficient Fine-Grained Pest Recognition via Enhanced MobileViT

Official implementation of **"Efficient Fine-Grained Pest Recognition via Enhanced MobileViT"**
(PRCV 2025, CCF-C, EI-indexed — [DOI: 10.1007/978-981-95-5699-1_5](https://doi.org/10.1007/978-981-95-5699-1_5)).

A lightweight variant of MobileViT-S for fine-grained, long-tailed insect pest classification on
IP102. Two modifications, no extra inference cost from the attention module:

1. **SimAM attention** (parameter-free) embedded into the intermediate MobileNetV2 blocks;
2. **Two-Stage FC Head** — dropout + batch normalisation + a 512-d bottleneck — to improve
   class separability under severe class imbalance.

| Model | Top-1 (%) | Macro-F1 (%) | Weighted-F1 (%) | Params (M) | Latency (ms/img) |
|---|---|---|---|---|---|
| MobileViT-S (baseline) | 64.66 | 56.14 | 64.01 | 5.02 | 0.683 |
| + SimAM | 68.07 | 61.12 | 67.56 | 5.02 | 0.814 |
| + Two-Stage FC Head | 68.33 | 61.49 | 67.64 | 5.32 | — |
| **Ours (SimAM + Two-Stage FC Head)** | **69.02** | **62.02** | **68.56** | **5.32** | **0.803** |

> +4.36 points Top-1 over the baseline with +0.3 M parameters. Latency measured at batch size 64
> on a single NVIDIA RTX 4090D; the model remains deployable on edge devices.

---

## 1. Dataset

### Source

[**IP102**](https://github.com/xpwu95/IP102) — *IP102: A Large-Scale Benchmark Dataset for Insect
Pest Recognition*, CVPR 2019 (Wu et al.).

- 75,222 images, **102 pest classes** under **8 crop super-classes** (rice, corn, wheat, beet,
  alfalfa, grape, citrus, mango)
- Collected in the wild: varied illumination, cluttered backgrounds, different life stages
- Strongly **long-tailed**, with high intra-class variation and high inter-class similarity
- Publicly available for research use — please follow the licence and citation terms of the
  original dataset

No external data, pseudo-labels or pre-training on IP102 itself is used, so results reflect what
the architecture learns from IP102 alone.

### Split protocol

We follow the **official IP102 classification protocol: 60% / 10% / 30%**:

| Split | Ratio | Images |
|---|---|---|
| Train | 60% | 45,095 |
| Validation | 10% | 7,508 |
| Test | 30% | 22,619 |
| **Total** | 100% | **75,222** |

Notes on the split logic:

- The split is **class-stratified and fixed** — the same official lists are used for every model in
  the paper, so the baseline and all ablations are directly comparable.
- **No re-sampling, over-sampling or class weighting** is applied. The long-tailed distribution is
  kept intact, which is why Macro-F1 is reported alongside Top-1.
- Validation (10%) is used only for early stopping and model selection; the test set (30%) is
  touched once, for the final numbers.
- All random seeds are fixed to **42** (`torch`, `cuda`, `numpy`, `random`, plus deterministic cuDNN).

### Expected on-disk layout

The data loader is folder-based: one directory per class, class name taken from `labels.json`.

```
data/IP102/
├── train/<class_name>/xxxxx.jpg
├── val/<class_name>/xxxxx.jpg
├── test/<class_name>/xxxxx.jpg
└── labels.json          # {"1": "<class_name>", "2": "<class_name>", ..., "102": "..."}
```

`labels.json` maps the **1-based** original IP102 class index to the class folder name; the loader
converts it to a 0-based index used by the classifier head (`num_classes=102`).

If you start from the raw IP102 release, reorganise the images into the three split folders above
according to the official split lists, keeping the original class folder names. Paths in
`train.py` / `Evaluate_IP102.py` are controlled by `--train_path`, `--val_path` and
`--label-json` (the evaluation script currently hard-codes its paths — see *Known issues*).

---

## 2. Environment

Verified with:

- Python 3.9+, PyTorch **2.0.0**, CUDA **11.8**
- Ubuntu 20.04, single NVIDIA **RTX 4090D**
- `torchvision`, `numpy`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `tensorboard`

```bash
conda create -n ip102 python=3.9 -y
conda activate ip102
pip install torch==2.0.0 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy scikit-learn pandas matplotlib seaborn tensorboard pillow
```

ImageNet-pretrained MobileViT-S weights (`mobilevit_s.pt`) are loaded from `type/` by default; the
classifier keys are dropped before loading so the head can be re-initialised for 102 classes.

---

## 3. Training

```bash
python train.py \
  --num_classes 102 \
  --epochs 100 \
  --batch-size 64 \
  --lr 0.001 \
  --train_path data/IP102/train \
  --val_path   data/IP102/val \
  --label-json data/IP102/labels.json \
  --weights type/mobilevit_s.pt \
  --device cuda:0
```

Training configuration:

| Item | Value |
|---|---|
| Input resolution | 224 × 224 |
| Augmentation | RandomResizedCrop, RandomHorizontalFlip |
| Normalisation | ImageNet mean/std |
| Optimiser | AdamW, lr 1e-3, weight decay 5e-4 |
| Scheduler | CosineAnnealingLR (T_max = epochs, eta_min 1e-6) |
| Epochs | 100 |
| Early stopping | patience 10, delta 0.001 (on validation accuracy) |
| Seed | 42 |

The best checkpoint is written to `weights/`; TensorBoard logs (loss / accuracy / lr per epoch) are
written to `runs/`.

---

## 4. Evaluation

```bash
python Evaluate_IP102.py
```

Outputs:

- Top-1 accuracy, Macro-F1, Weighted-F1
- `test_classification_report.csv` — per-class precision / recall / F1
- `test_confusion_matrix.png` — 102 × 102 confusion matrix
- average inference time per batch and throughput (images/s)

Before running, set `test_root`, `label_json` and the checkpoint path at the top of `main()`.

---

## 5. Repository structure

```
├── train.py                     # training / validation loop, early stopping
├── Evaluate_IP102.py            # test-set evaluation, report + confusion matrix
├── model.py                     # MobileViT-S baseline
├── model_config_SimAM.py        # MobileViT-S + SimAM configuration
├── model_lib/                   # model definitions used by the evaluation script
├── module/                      # SimAM and attention blocks
├── DataLoader/IP102_dataset.py  # folder-based IP102 dataset + collate_fn
├── tools/                       # helper scripts (data preparation, latency benchmark)
├── utils.py                     # train_one_epoch / evaluate
├── transformer.py               # transformer encoder blocks
├── unfold_test.py               # patch-unfold sanity check
└── type/                        # ImageNet-pretrained MobileViT-S weights
```

---

## 6. Known issues / TODO

- `Evaluate_IP102.py` currently applies `RandomResizedCrop` to the **test** set, which makes the
  reported numbers slightly non-deterministic. For a fully reproducible evaluation, replace it with
  `Resize(256) + CenterCrop(224)`; the numbers in the paper were produced with the deterministic
  pipeline.
- Paths in `Evaluate_IP102.py` are hard-coded (originally run on a remote GPU box); move them to
  `argparse` before reuse.
- The training script still saves the checkpoint under a legacy filename
  (`model_SimAM_CdAtt_IP102.pth`); rename if you prefer.
- A `.gitignore` is not yet included (`.idea/`, `__pycache__/`, `*.pth` should be excluded).

---

## 7. Citation

```bibtex
@inproceedings{liu2025pest,
  title     = {Efficient Fine-Grained Pest Recognition via Enhanced MobileViT},
  author    = {Liu, Lizhao and Zhao, Jiahao},
  booktitle = {Pattern Recognition and Computer Vision (PRCV)},
  year      = {2025},
  doi       = {10.1007/978-981-95-5699-1_5}
}
```

If you use the IP102 dataset, please also cite the original dataset paper:

```bibtex
@inproceedings{wu2019ip102,
  title     = {IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition},
  author    = {Wu, Xiaoping and Zhan, Chi and Lai, Yukun and Cheng, Ming-Ming and Yang, Jufeng},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2019}
}
```

---

## 8. Contact

Jiahao Zhao — z1581117982@gmail.com
Questions about the code or the paper are welcome; please open an issue for reproducibility problems.
