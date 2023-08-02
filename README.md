# Partial Ordered Gromov-Wasserstein

## For Developers

### Setup

#### Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


## Run POGW experiments

* Moving Digits Alignment Experiments

```bash
python -m src.experiments.digit_moving.alignment
```

Read src/experiments/digit_moving/alignment.py for more details.

* Multi-feature Weizmann Classification K-NN Experiments

Example :

```bash
python -m src.experiments.multi_features_weizmann.knn_eval \
                    --test_size 0.5 \
                    --random_outlier \
                    --metric euclidean \
                    --m 0.75 \
                    --reg 1000 \
                    --algo parial-order-gromov
```

Read src/experiments/multi_features_weizmann/knn_eval.py for more details.

* Note : to run GDTW follow instructions in [this repo](https://github.com/samcohen16/Aligning-Time-Series)
