# AVE — Audio-Visual Event Localization

CoLA for Audio-Visual Event Localization on the **AVE** dataset.

```
ave/
├── CoLA/   # our method (cross-modal LoRA)
└── LoRA/   # LoRA baseline
```

Both variants share the same layout:

```
CoLA/
├── main_trans.py        # training / evaluation entry point
├── train.sh             # example training command
├── eval.sh              # example evaluation command
├── base_options.py      # all command-line arguments and defaults
├── dataloader.py        # AVE data loading
├── nets/                # model definition
├── SSLAM/               # audio backbone (SSLAM)
├── custom_timm/         # vendored timm (no install needed)
└── loralib/             # LoRA / CoLA adaptation modules
```

## 1. Environment

```bash
pip install -r requirements.txt
```

## 2. Dataset

Download the **AVE** dataset from the official release:
[AVE-ECCV18](https://github.com/YapengTian/AVE-ECCV18).
Extract video frames and audio, then arrange them as:

```
AVE_Dataset/
├── video_frames/
│   └── VIDEO_NAME/
│       ├── 0001.jpg
│       └── ...
└── raw_audio/
    └── VIDEO_NAME.wav
```

> Preprocessed frames/audio are also available from
> [LAVISHData](https://huggingface.co/datasets/genjib/LAVISHData/).

### SSLAM audio backbone (required)

This task uses the **SSLAM** audio backbone ([SSLAM](https://github.com/ta012/SSLAM)).
Download its repository and finetuned checkpoint, then update the placeholder
(`/path/to/...`) paths in the two files used during training:

| File | Variable(s) to set |
|------|--------------------|
| `main_trans.py` (training entry) | `fairseq_path`, `user_dir_path` (SSLAM folder) |
| `SSLAM/sslam_backbone.py` | `fairseq_path`, `checkpoint_dir` (SSLAM `.pt`), `model_dir` (SSLAM folder) |

## 3. Train & evaluate

Edit the dataset paths at the top of `train.sh` (`VIDEO_FOLDER`, `AUDIO_FOLDER`), then:

```bash
bash train.sh      # train
bash eval.sh       # evaluate
```

All hyper-parameters and paths can be overridden via the flags in `base_options.py`
(e.g. `--audio_folder`, `--video_folder`, `--lr`, `--lora_r`, `--lora_alpha`).

Run the same commands inside `CoLA/` for our method, or `LoRA/` for the baseline.
