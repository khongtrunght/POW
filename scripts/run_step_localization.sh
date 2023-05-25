#! /bin/bash
for reg in 0.1 0.15 0.2 0.25 0.3 1 2 3; do
    python -m src.experiments.step_localization.evaluate  --algorithm=POW-reg --keep_percentile 0.4 --reg 0.25 --reg2 $reg --use_unlabeled --metric cosine
    # python -m src.experiments.step_localization.evaluate  --algorithm=POW --keep_percentile 0.4 --reg $reg --reg2 0 --use_unlabeled
done
