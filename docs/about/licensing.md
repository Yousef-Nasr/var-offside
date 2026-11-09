# Licensing

## VAR-Offside

VAR-Offside is licensed under the **GNU Affero General Public License v3.0 or later** ([LICENSE](https://github.com/Yousef-Nasr/var-offside/blob/main/LICENSE)).

In short: you may use, study, change and share it. If you distribute it, or let people use a modified version over a network (for example a hosted web app), you must offer them the source code of your version under the same licence.

The licence follows from the core dependency. Detection uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), which is AGPL-3.0, and VAR-Offside's detector is a fine-tuned YOLO model. Ultralytics also sells an [enterprise licence](https://www.ultralytics.com/license) if AGPL doesn't suit your use.

## Third-party code and models

VAR-Offside doesn't bundle any of these. You download each one from its source, under its own licence.

| Component | Licence | Used for | How you get it |
|---|---|---|---|
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | AGPL-3.0 | Detection and pose | pip dependency |
| `best_tuned.pt` | AGPL-3.0 | Football detector, fine-tuned from YOLO | VAR-Offside release |
| `yolo11x-pose.pt`, `yolo11l.pt` | AGPL-3.0 | Pose; person detection for 3D | Ultralytics release |
| [PnLCalib](https://github.com/mguti97/PnLCalib) code and weights | GPL-2.0 | Automatic calibration | Clone into `third_party/`, weights from its release |
| [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) code and checkpoints | SAM License (Meta) | 3D player bodies | Clone into `third_party/`; gated download from Hugging Face |
| [Fast-SAM-3D-Body](https://github.com/yangtiming/Fast-SAM-3D-Body) | MIT | Faster 3D bodies | Clone into `third_party/` |
| [torchvision](https://github.com/pytorch/vision) ResNet-50 weights | BSD-3-Clause | Team embeddings | Downloaded by torchvision |

The **SAM License** isn't an open-source licence and has its own use restrictions. Read it before you use the 3D bodies, and especially before any commercial use. The rest of VAR-Offside works without it.

The full list, including Python packages and front-end libraries, is in [THIRD_PARTY_NOTICES.md](https://github.com/Yousef-Nasr/var-offside/blob/main/THIRD_PARTY_NOTICES.md).

## Citing

If you use VAR-Offside in research, please cite it with the metadata in [CITATION.cff](https://github.com/Yousef-Nasr/var-offside/blob/main/CITATION.cff) (GitHub shows a "Cite this repository" button), and cite PnLCalib and SAM 3D Body when you use them.

This isn't legal advice. Check each licence yourself for your use case.
