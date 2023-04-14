from src.experiments.step_localization.evaluate import main as evaluate_main
from argparse import Namespace
import pytest

def test_evaluate_main():

    args = Namespace(
        algorithm="POW",
        dataset="COIN",
        distance="inner",
        drop_cost="logit",
        keep_percentile=0.3,
        name="",
        reg=3,
        use_unlabeled=True,
    )

    evaluate_main(args)
