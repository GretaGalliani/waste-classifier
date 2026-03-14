# Waste Image Classifier

A deep learning image classifier (PyTorch) that identifies which recycling bin each object belongs to.

## Classes

| Class     | Bin              |
|-----------|------------------|
| cardboard | Paper/Cardboard  |
| glass     | Glass            |
| metal     | Metal            |
| paper     | Paper            |
| plastic   | Plastic          |
| trash     | General waste    |


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

## Project Phases

- [x] **Phase 1** — Dataset and preprocessing
- [ ] **Phase 2** — Model (transfer learning)
- [ ] **Phase 3** — Training and optimization
- [ ] **Phase 4** — Evaluation and error analysis
- [ ] **Phase 5** — Interactive web app
