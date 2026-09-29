# Fraud Management AI

Machine-learning experiments and preprocessing utilities for fraud detection.

## Setup

This project uses Python 3.11 and Poetry:

```bash
poetry install
poetry run python -m ipykernel install --user --name fraud-management
```

## Project layout

- `preprocessing/`: reusable feature preprocessing utilities
- `*.ipynb`: exploratory analysis and modeling notebooks
- `evaluate.py`: model evaluation entry point

The dependency lock file is committed to keep environments reproducible.