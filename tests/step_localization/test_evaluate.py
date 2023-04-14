from src.experiments.step_localization.evaluate import main as evaluate_main
from argparse import Namespace
import pytest

def test_evaluate_main_POW():

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

    result = evaluate_main(args)
    assert result['accuracy'] == pytest.approx(52.3, 0.1)
    assert result['iou'] == pytest.approx(19.8, 0.1)

def test_evaluate_main_DropDTW():

    args = Namespace(
        algorithm="DropDTW",
        dataset="COIN",
        distance="inner",
        drop_cost="logit",
        keep_percentile=0.3,
        name="",
        use_unlabeled=True,
    )

    result = evaluate_main(args)
    assert result['accuracy'] == pytest.approx(51.2, 0.1)
    assert result['iou'] == pytest.approx(23.6, 0.1)
