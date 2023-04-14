import argparse
import random

import numpy as np
import ot

np.random.seed(42)
random.seed(42)
sklearn_seed = 0

from sklearn.metrics import accuracy_score
from tqdm import tqdm

from config.config import WEI_PATH, logger
from src.dp.exact_dp import drop_dtw_distance, dtw_distance
from src.experiments.weizmann.dataset import WeisDataset
from src.experiments.weizmann.utils import add_outlier
from src.pow.pow import pow_distance


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size", type=float, default=0.5)
    parser.add_argument("--outlier_ratio", type=float, default=0.1)
    parser.add_argument("--metric", type=str)
    parser.add_argument("--m", type=float, default=0.1)
    parser.add_argument("--reg", type=int, default=1)
    parser.add_argument("--distance", type=str, default="euclidean")
    args = parser.parse_args()
    return args


def main(args):
    logger.info(f"Args: {args}")
    weis_dataset = WeisDataset.from_folder(WEI_PATH, test_size=args.test_size)
    X_train = [weis_dataset.get_sequence(idx) for idx in weis_dataset.train_idx]
    X_test = [weis_dataset.get_sequence(idx) for idx in weis_dataset.test_idx]

    X_test = list(
        map(lambda x: add_outlier(x, outlier_ratio=args.outlier_ratio), X_test)
    )

    y_train = (weis_dataset.get_label(idx) for idx in weis_dataset.train_idx)
    y_test = (weis_dataset.get_label(idx) for idx in weis_dataset.test_idx)
    y_train = np.array(list(y_train))
    y_test = np.array(list(y_test))

    fn_dict = {
        "pow": pow_distance,
        "dtw": dtw_distance,
        "drop_dtw": drop_dtw_distance,
    }

    train_size = len(X_train)
    test_size = len(X_test)

    result = np.zeros((train_size, test_size))
    for train_idx in tqdm(range(train_size)):
        for test_idx in tqdm(range(test_size), leave=False):
            M = ot.dist(X_train[train_idx], X_test[test_idx], metric=args.distance)
            if args.metric == "pow":
                result[train_idx, test_idx] = fn_dict[args.metric](
                    M, m=args.m, reg=args.reg
                )
            elif args.metric == "drop_dtw":
                result[train_idx, test_idx] = fn_dict[args.metric](
                    M, keep_percentile=args.m
                )
            else:
                result[train_idx, test_idx] = fn_dict[args.metric](M)

    y_pred = y_train[np.argmin(result, axis=0)]
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
