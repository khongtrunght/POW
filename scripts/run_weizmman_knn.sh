#! /bin/bash
for metric in dtw drop_dtw pow; do
    for k in 1 3 5 7; do
        python -m src.experiments.weizmann.knn_eval --test_size 0.5 --outlier_ratio 0.3 --metric $metric --m 0.7 --reg 1 --distance cityblock --k $k
    done
done
