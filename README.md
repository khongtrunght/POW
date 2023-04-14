# POW

# For Developers

## Setup

### Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements/dev.txt
pre-commit install
```

# Run experiments

* Step localization experiments

Example:

```bash
python src/evaluate.py --algorithm=POW --keep_percentile 0.3 --reg 3 --use_unlabeled
```

Read src/evaluate.py for more details.
