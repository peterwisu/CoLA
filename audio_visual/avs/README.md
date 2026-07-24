# AVS — Audio-Visual Segmentation

CoLA for Audio-Visual Segmentation on the **AVSBench** (Single-source / S4) dataset.

```
avs/
├── CoLA/   # our method (cross-modal LoRA)
└── LoRA/   # LoRA baseline
```

Both variants share the same layout:

```
CoLA/
├── train.py / train.sh    # training entry point + example command
├── test.py  / test.sh     # evaluation entry point + example command
├── config.py              # dataset paths and data config
├── base_options.py        # command-line arguments
├── dataloader.py          # AVSBench data loading
├── model/                 # segmentation model (PVT / ResNet backbone)
├── SSLAM/                 # audio backbone (SSLAM)
├── custom_timm/           # vendored timm (no install needed)
└── loralib/               # LoRA / CoLA adaptation modules
```

## 1. Environment

```bash
pip install -r requirements.txt
```

## 2. Dataset

Download the **AVSBench** dataset (Single-source / S4 subset) from the official source:
[AVSBench](https://github.com/OpenNLPLab/AVSBench). Expected layout:

```
avsbench_data/
└── Single-source/
    ├── s4_meta_data.csv
    └── s4_data/
        ├── visual_frames/
        ├── audio_log_mel/
        ├── audio_wav/
        └── gt_masks/
```

Set these paths in [`config.py`](CoLA/config.py) (lines `cfg.DATA.*`).

### SSLAM audio backbone (required)

This task uses the **SSLAM** audio backbone ([SSLAM](https://github.com/ta012/SSLAM)).
Download its repository and finetuned checkpoint, then update the placeholder
(`/path/to/...`) paths in the two files used during training:

| File | Variable(s) to set |
|------|--------------------|
| `train.py` (training entry) | `fairseq_path`, `user_dir_path` (SSLAM folder) |
| `SSLAM/sslam_backbone.py` | `fairseq_path`, `checkpoint_dir` (SSLAM `.pt`), `model_dir` (SSLAM folder) |

## 3. Train & evaluate

```bash
bash train.sh                       # train
# set WEIGHT=/path/to/checkpoint.pth in test.sh, then:
bash test.sh                        # evaluate
```

Key flags (see `train.sh`): `--visual_backbone` (`pvt` or `resnet`), `--lr`,
`--train_batch_size`, `--use_lora`, `--lora_r`, `--lora_alpha`.

Run the same commands inside `CoLA/` for our method, or `LoRA/` for the baseline.
