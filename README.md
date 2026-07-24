<div align="center">
  <img width="128" height="128" alt="CoLA" src="https://github.com/user-attachments/assets/7325a5c2-4b70-4896-97f8-6658a4f3caa7" />

  # 🥤 CoLA: Cross-Modal Low-rank Adaptation for Multimodal Downstream Tasks

  [![Conference](https://img.shields.io/badge/ICML-2026-blue)](https://icml.cc/virtual/2026/poster/65985)
  [![arXiv](https://img.shields.io/badge/arXiv-2604.03314-b31b1b.svg)](https://arxiv.org/abs/2604.03314)

  📄 [**Paper**](https://arxiv.org/abs/2604.03314) &nbsp;|&nbsp; 🔗 [**ICML 2026 Poster**](https://icml.cc/virtual/2026/poster/65985)
</div>

Official implementation of **CoLA: Cross-Modal Low-rank Adaptation for Multimodal
Downstream Tasks** (ICML 2026). CoLA is a parameter-efficient adaptation method that
injects cross-modal low-rank updates to better fuse information across modalities.
We evaluate CoLA on three multimodal downstream tasks:
**Audio-Visual Event Localization (AVE)**, **Audio-Visual Segmentation (AVS)**, and
**Vision-Language visual grounding (VL)**.

## 📋 Table of Contents

- [🖼️ Poster](#-poster)
- [📦 Repository Structure](#-repository-structure)
- [🚀 Tasks](#-tasks)
- [🙏 Acknowledgements](#-acknowledgements)
- [📜 Citation](#-citation)

## 🖼️ Poster

<div align="center">
  <img src="https://icml.cc/media/PosterPDFs/ICML%202026/65985.png" width="850" alt="CoLA — ICML 2026 poster" />
</div>

## 📦 Repository Structure

```
.
├── audio_visual/
│   ├── ave/                  # Audio-Visual Event Localization (AVE)
│   │   ├── CoLA/             #   our method (cross-modal LoRA)
│   │   └── LoRA/             #   LoRA baseline
│   └── avs/                  # Audio-Visual Segmentation (AVS)
│       ├── CoLA/             #   our method
│       └── LoRA/             #   LoRA baseline
└── vision_language/          # Vision-Language visual grounding (VL)
    ├── CoLA/                 #   our method
    └── LoRA/                 #   LoRA baseline
```

For every task we provide two variants:

- **`CoLA/`** — our proposed cross-modal low-rank adaptation.
- **`LoRA/`** — the standard LoRA baseline used for comparison.

The two variants share the same structure and data; they differ only in the adaptation
module. Each variant is self-contained and can be run independently.

## 🚀 Tasks

Head to the per-task README for dataset download, folder layout, and run commands.

| Task | Description | Setup & instructions |
|------|-------------|----------------------|
| **AVE** | Audio-Visual Event Localization | [audio_visual/ave/README.md](audio_visual/ave/README.md) |
| **AVS** | Audio-Visual Segmentation | [audio_visual/avs/README.md](audio_visual/avs/README.md) |
| **VL**  | Vision-Language visual grounding | [vision_language/README.md](vision_language/README.md) |

## 🙏 Acknowledgements

Our code is primarily based on [LAVISH](https://github.com/GenjiB/LAVISH) (AVE & AVS) and
[EEVG](https://github.com/chenwei746/EEVG) (VL), with the audio backbone adapted from
[SSLAM](https://github.com/ta012/SSLAM) and the low-rank adaptation modules built on
[LoRA](https://github.com/microsoft/LoRA). We thank the authors for releasing their code.

## 📜 Citation

If you find CoLA useful in your research, please consider citing:

```bibtex

```
