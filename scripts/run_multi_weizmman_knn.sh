#! /bin/bash
for algo in parial-order-gromov gromov-dtw gromov partial-gromov; do
    for i in {0..4}; do
        python -m src.experiments.multi_features_weizmann.knn_eval \
            --test_size 0.5 \
            --outlier_ratio 0.1 \
            --metric euclidean \
            --m 0.7 \
            --reg 10 \
            --algo $algo
    done
done

