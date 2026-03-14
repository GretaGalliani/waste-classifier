# Waste Image Classifier

A deep learning image classifier (PyTorch) that identifies which recycling bin each object belongs to.

## Classes

| Class     | Bin             |
|-----------|-----------------|
| cardboard | Paper/Cardboard |
| glass     | Glass           |
| metal     | Metal           |
| paper     | Paper           |
| plastic   | Plastic         |
| trash     | General waste   |


## Project Structure

```
waste-classifier/
├── data/
│   ├── raw/              # Original dataset (from Kaggle)
│   └── split/            # Train/val/test split (generated)
│       ├── train/
│       ├── val/
│       └── test/
├── outputs/
│   └── eda/              # Exploratory data analysis plots
├── phase1_dataset.py     # Phase 1: Dataset and preprocessing
├── pyproject.toml
├── poetry.lock
└── README.md
```

## Setup

### Prerequisites

- Python 3.11
- [Poetry](https://python-poetry.org/docs/#installation)

### Install dependencies

```bash
poetry install
```

### Kaggle API

The dataset is downloaded from Kaggle via the official API.

1. Sign in to [kaggle.com](https://www.kaggle.com) and go to **Account → Settings → API**.
   Copy your **username** and **API key**.

2. Create the credentials file manually:

   ```bash
   mkdir -p ~/.kaggle
   echo '{"username":"YOUR_USERNAME","key":"YOUR_API_KEY"}' > ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. Verify it works:

   ```bash
   poetry run kaggle datasets list
   ```

3. Download the dataset:

   ```bash
   poetry run python phase1_dataset.py --download
   ```

   This will download and extract the
   [Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
   dataset into `data/raw/`.

## Phase 1 — Dataset and Preprocessing

`phase1_dataset.py` handles the full data pipeline. Each step can be run independently or all at once.

### Run all steps

```bash
poetry run python phase1_dataset.py --all
```

### Run individual steps

```bash
# 1. Download dataset from Kaggle into data/raw/
poetry run python phase1_dataset.py --download

# 2. Split into train / val / test (70 / 15 / 15)
poetry run python phase1_dataset.py --split

# 3. Generate EDA plots into outputs/eda/
poetry run python phase1_dataset.py --eda
```

### Output

| Step | Output |
|------|--------|
| `--download` | `data/raw/{class}/` — raw images per class |
| `--split` | `data/split/train/`, `val/`, `test/` — stratified folders |
| `--eda` | `outputs/eda/class_distribution.png`, `sample_images.png` |

### Using the DataLoader in code

```python
from phase1_dataset import get_dataloaders

loaders = get_dataloaders(batch_size=32)
for images, labels in loaders["train"]:
    ...  # images: (B, 3, 224, 224), labels: (B,)
```

## Project Phases

- [x] **Phase 1** — Dataset and preprocessing
- [ ] **Phase 2** — Model (transfer learning)
- [ ] **Phase 3** — Training and optimization
- [ ] **Phase 4** — Evaluation and error analysis
- [ ] **Phase 5** — Interactive web app
