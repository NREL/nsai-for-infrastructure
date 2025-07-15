# `nsai_experiments`

## Installation for Development
After cloning the repository, create a virtual environment in the repository root, activate it, and install the project in editable mode with its development dependencies:
```bash
python -m venv .venv  # using Python >= 3.10
source .venv/bin/activate
pip install -e '.[dev]'
```

If you have had to install an old version of PyTorch (e.g., 2.2), you may need to manually downgrade to NumPy 1:
```bash
pip install 'numpy<2'
```

If possible, prefer a newer version of PyTorch (e.g., 2.6).

## Running Tests
Simply invoke `pytest`:
```bash
pytest
```
