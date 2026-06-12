# PCB Defect Detection - SolderNet Implementation

ConvNeXt-Tiny based PCB defect classifier with from-scratch Grad-CAM explanations for DeepPCB solder joint defect crops.

## Reference

This project is inspired by:

> Hayden Gunraj, Paul Guerrier, Sheldon Fernandez, and Alexander Wong. "SolderNet: Towards Trustworthy Visual Inspection of Solder Joints in Electronics Manufacturing Using Explainable Artificial Intelligence." AAAI Conference on Artificial Intelligence, 2023. DOI: [10.1609/aaai.v37i13.26858](https://doi.org/10.1609/aaai.v37i13.26858). arXiv: [2211.10274](https://arxiv.org/abs/2211.10274).

The demo adapts the explainable inspection idea to the [DeepPCB](https://github.com/tangsanli5201/DeepPCB) dataset by cropping defect annotations into a 7-class classification task:

`open`, `short`, `mousebite`, `spur`, `copper`, `pin-hole`, and `good`.

## Architecture

```text
DeepPCB template/test pairs + bbox annotations
                |
                v
      data/download_data.py
  defect crops + non-overlap good crops
                |
                v
 train.csv / val.csv / test.csv
                |
                v
 ImageNet normalization + augmentation
                |
                v
 ConvNeXt-Tiny backbone (ImageNet weights)
   - first two stages frozen for warm-up
   - all stages unfrozen for fine-tuning
                |
                v
 Linear classifier head, 7 classes
                |
        +-------+--------+
        |                |
        v                v
 class prediction   Grad-CAM on features[7][2].block[0]
```

## Repository Layout

```text
pcb-defect-detection/
├── README.md
├── requirements.txt
├── data/
│   └── download_data.py
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── gradcam.py
├── app/
│   └── demo.py
├── checkpoints/
└── results/
    └── metrics.json
```

## Setup

Python 3.10+ is required.

```bash
git clone <your-repo-url>
cd pcb-defect-detection
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare DeepPCB Crops

The data script clones DeepPCB, finds matching test images and annotation files, crops defect bounding boxes, samples non-overlapping `good` patches, and writes 70/15/15 CSV splits with `filepath,label` columns.

```bash
python data/download_data.py
```

Outputs:

- `data/raw/DeepPCB/` - cloned source dataset
- `data/processed/crops/` - crop-level classification dataset
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`

## Train

```bash
python src/train.py \
  --train-csv data/processed/train.csv \
  --val-csv data/processed/val.csv \
  --epochs 30
```

Training uses:

- `torchvision.models.convnext_tiny` with ImageNet weights
- AdamW, learning rate `1e-4`, weight decay `1e-2`
- cosine annealing over 30 epochs
- class-weighted cross entropy
- horizontal/vertical flips, rotation, color jitter, ImageNet normalization
- best checkpoint selected by validation macro F1

The best checkpoint is written to:

```text
checkpoints/best_model.pth
```

Large checkpoint files are intentionally gitignored. If you already have a trained checkpoint, place it at `checkpoints/best_model.pth`.

## Evaluate

```bash
python src/evaluate.py \
  --test-csv data/processed/test.csv \
  --checkpoint checkpoints/best_model.pth
```

Outputs:

- `results/metrics.json`
- `results/confusion_matrix.png`

## Run the Streamlit Demo

```bash
streamlit run app/demo.py
```

The app has two modes:

1. **Upload Image** - predicts a class, displays confidence scores, and overlays Grad-CAM.
2. **Run on Test Set Sample** - samples 9 test crops, shows true/predicted labels, and displays saved metrics.

If `checkpoints/best_model.pth` is missing, the app shows a warning and runs with random weights so the UI remains explorable.

## Results

Populate this table after training and evaluation. The committed placeholder metrics are zeros until `python src/evaluate.py` is run.

| Class | F1 |
| --- | ---: |
| open | 0.000 |
| short | 0.000 |
| mousebite | 0.000 |
| spur | 0.000 |
| copper | 0.000 |
| pin-hole | 0.000 |
| good | 0.000 |
| **Overall accuracy** | **0.000** |

## Demo Screenshots

Upload image mode:

```text
[Streamlit screenshot placeholder: uploaded PCB image with prediction and Grad-CAM overlay]
```

Test sample mode:

```text
[Streamlit screenshot placeholder: 3x3 test image grid and metrics panel]
```

## License

MIT
