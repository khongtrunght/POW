import argparse
import random

import numpy as np
import ot

np.random.seed(42)
random.seed(42)
sklearn_seed = 0

from sklearn.metrics import accuracy_score
from tqdm import tqdm

from config.config import logger
from src.dp.exact_dp import drop_dtw_distance, dtw_distance
from src.experiments.ucr.utils import get_train_test_data, random_add_noise_with_seed
from src.pow.pow import pow_distance
from src.utils.knn_utils import knn_classifier_from_distance_matrix


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outlier_ratio", type=float, default=0.1)
    parser.add_argument("--metric", type=str)
    parser.add_argument("--m", type=float, default=0.1)
    parser.add_argument("--reg", type=int, default=1)
    parser.add_argument("--distance", type=str, default="euclidean")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()
    return args


def main(args):
    logger.info(f"Args: {args}")
    X_train, y_train, X_test, y_test = get_train_test_data(dataset=args.dataset)
    X_train_mean = np.mean(X_train, axis=0)
    X_train_std = np.std(X_train, axis=0)
    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std
    fn_dict = {
        "pow": pow_distance,
        "dtw": dtw_distance,
        "drop_dtw": drop_dtw_distance,
    }

    train_size, test_size = X_train.shape[0], X_test.shape[0]
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    X_test_outlier = random_add_noise_with_seed(X_test, args.outlier_ratio, 42)
    logger.info("X_test_outlier shape: {}".format(X_test_outlier.shape))
    X_test = X_test_outlier

    result = np.zeros((test_size, train_size))
    for train_idx in tqdm(range(train_size)):
        for test_idx in tqdm(range(test_size), leave=False):
            x_tr = X_train[train_idx].reshape(-1, 1)
            x_te = X_test[test_idx].reshape(-1, 1)
            M = ot.dist(x_tr, x_te, metric=args.distance)
            if args.metric == "pow":
                distance = fn_dict[args.metric](M, m=args.m, reg=args.reg)
            elif args.metric == "drop_dtw":
                distance = fn_dict[args.metric](M, keep_percentile=args.m)
            else:
                distance = fn_dict[args.metric](M)
            if distance == np.inf:
                distance = np.max(result)
            result[test_idx, train_idx] = distance

    y_pred = knn_classifier_from_distance_matrix(
        distance_matrix=result,
        k=args.k,
        labels=y_train,
    )
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
