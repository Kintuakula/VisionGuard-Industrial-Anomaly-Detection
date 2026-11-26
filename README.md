# Self-Supervised Anomaly Detection for Industrial Visual Inspection

Beginner-friendly VS Code starter project using Python, PyTorch, a pretrained ResNet18 feature extractor, and a PatchCore-style normal-feature memory bank.

## 1. Install
Open this folder in VS Code. In the VS Code terminal:

```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py` does not work, try `python` instead. In VS Code, select the `.venv` Python interpreter.

## 2. Download the MVTec AD dataset
Download MVTec AD from its official project page and extract it. Copy one category folder (for example `bottle`) into `data/MVTec_AD/` so that the structure is:

```text
data/
  MVTec_AD/
    bottle/
      train/
        good/
          image1.png
      test/
        good/
        broken_large/
        broken_small/
      ground_truth/
        broken_large/
        broken_small/
```

Folder names for defect types vary by category. Keep the original dataset folder names.

## 3. Train the normal memory bank
```bash
python main.py train --data data/MVTec_AD/bottle --output outputs/bottle
```

Only images under `train/good` are used to build the normal memory bank.

## 4. Test and create heatmaps
```bash
python main.py test --data data/MVTec_AD/bottle --output outputs/bottle
```

Results are saved under `outputs/bottle/results/`. The script prints image-level AUROC when both normal and anomalous test images are present, and pixel-level AUROC when usable ground-truth masks are available.

## Notes
- This is a practical PatchCore-style starter, not an exact reproduction of every PatchCore paper detail.
- The pretrained ResNet18 weights may download on first run, so internet access is needed once.
- Use a small category first. CPU works but may be slow; a compatible NVIDIA GPU can help.
- This project does not require SQL or MongoDB. Images and a saved `.pt` feature memory bank are sufficient.
