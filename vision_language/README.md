# VL — Vision-Language Visual Grounding

CoLA for visual grounding (referring expression comprehension & segmentation),
built on the [EEVG](https://github.com/chenwei746/EEVG) framework.

```
vision_language/
├── CoLA/   # our method (cross-modal LoRA)
└── LoRA/   # LoRA baseline
```

Both variants share the same layout:

```
CoLA/
├── train.py / train.sh    # training entry point + example commands
├── eval.py  / test.sh     # evaluation entry point + example commands
├── engine.py              # train / eval loops
├── datasets/              # dataset loading (RefCOCO, Flickr30K, VG, ...)
├── models/                # visual model, language model, decoder
└── loralib/               # LoRA / CoLA adaptation modules
```

## 1. Environment

- Python 3.8, PyTorch 2.0.1 + cu117
```bash
pip install -r requirements.txt
```

## 2. Dataset

Download the grounding images and place them under `./ln_data`:
**RefCOCO / RefCOCO+ / RefCOCOg**, **Flickr30K Entities**, **Visual Genome**.

```
ln_data/
├── flickr30k/
├── other/images/mscoco/images/train2014/
└── visual-genome/
```

Download the data labels and place them under `./mask_data`, and the pretrained
backbones under `./checkpoints`:

- Labels: see [EEVG](https://github.com/chenwei746/EEVG)
- [ViTDet](https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_b/f325358525/model_final_435fa9.pkl)
- [SwinT](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth)

## 3. Train & evaluate

```bash
bash train.sh      # training commands for each dataset
bash test.sh       # evaluation commands for each split / dataset
```

Both scripts use `torch.distributed.launch` for multi-GPU runs. Key flags:
`--backbone` (`ViTDet` / `SwinT`), `--dataset` (e.g. `gref_umd`, `unc`, `unc+`),
`--is_segment`, `--vl_enc_layers`, and LoRA/CoLA options in `loralib/`.

Run the same commands inside `CoLA/` for our method, or `LoRA/` for the baseline.
