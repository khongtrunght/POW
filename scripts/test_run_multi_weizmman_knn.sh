for algo in parial-order-gromov; do
    for m in 0.7 0.8 0.9; do
        python -m src.experiments.multi_features_weizmann.knn_eval \
            --test_size 0.5 \
            --random_outlier \
            --m $m \
            --metric euclidean \
            --reg 10 \
            --algo $algo
    done
done