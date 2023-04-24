#! /bin/bash
for metric in drop_dtw dtw softdtw pow dtw opw; do
    for k in 1 3 5 7; do
        for seed in 1 2 3 4 5; do
            for distance in cityblock cosine ; do
                python -m src.experiments.weizmann.knn_eval --test_size 0.5 --outlier_ratio 0.3 --metric $metric --m 0.7 --reg 1 --distance $distance --k $k --seed $seed
            done
        done
    done
done
