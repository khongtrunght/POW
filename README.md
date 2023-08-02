# POW

## For Developers

### Setup

#### Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


## Run POW experiments

* Moving Digits Alignment Experiments

```bash
python -m src.experiments.digit_moving.alignment
```

* Weizmann classification knn experiments

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


* UCR classification k-nn experiments

Example

```bash
python -m src.experiments.ucr.knn_eval --dataset=Chinatown --outlier_ratio 0.1 --metric pow  --m 0.9 --reg 1 --distance euclidean --seed 1
```

* Note : to run GDTW follow instructions in [this repo](https://github.com/samcohen16/Aligning-Time-Series)
