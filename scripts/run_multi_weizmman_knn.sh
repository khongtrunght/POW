# #! /bin/bash
for algo in parial-order-gromov; do
    for m in 0.6 0.7 0.8 0.9; do
        for reg in 0.1 1 10 50 100; do
            for i in {0..4}; do
                python -m src.experiments.multi_features_weizmann.knn_eval \
                    --test_size 0.5 \
                    --random_outlier \
                    --metric euclidean \
                    --m $m \
                    --reg $reg \
                    --algo $algo
            done
        done
    done
done