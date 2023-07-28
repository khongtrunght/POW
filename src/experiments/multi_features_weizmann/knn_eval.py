import argparse
import os
import json
import numpy as np
import ot
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

from config.config import MULTI_WEI_PATH, logger
from src.experiments.multi_features_weizmann.dataset import WeisDataset
from src.experiments.multi_features_weizmann.utils import add_outlier
from src.pogw.pogw import partial_order_gromov_wasserstein
# from src.pogw.pogw_2 import partial_order_gromov_wasserstein
from src.utils.knn_utils import knn_classifier_from_distance_matrix
from src.gdtw.GDTW import gromov_dtw


def partial_order_gromov_dist(x1, x2, order_reg, m=None, metric="euclidean"):
    C1 = ot.dist(x1, x1, metric=metric)
    C2 = ot.dist(x2, x2, metric=metric)
    # C1 = C1 / C1.max()
    # C2 = C2 / C2.max()
    C1 = C1 / C1.mean()
    C2 = C2 / C2.mean()

    p = ot.unif(C1.shape[0])
    q = ot.unif(C2.shape[0])

    dist = partial_order_gromov_wasserstein(
        C1, C2, p, q, m=m, order_reg=order_reg, return_dist=True
    )

    return dist


def partial_gromov_dist(x1, x2, m=None, metric="euclidean"):
    C1 = ot.dist(x1, x1, metric=metric)
    C2 = ot.dist(x2, x2, metric=metric)
    C1 = C1 / C1.max()
    C2 = C2 / C2.max()
    p = ot.unif(C1.shape[0])
    q = ot.unif(C2.shape[0])
    dist = ot.partial.partial_gromov_wasserstein2(C1, C2, p, q, m=m)
    return dist


def gromov_dist(x1, x2, metric="euclidean"):
    C1 = ot.dist(x1, x1, metric=metric)
    C2 = ot.dist(x2, x2, metric=metric)
    C1 = C1 / C1.max()
    C2 = C2 / C2.max()
    p = ot.unif(C1.shape[0])
    q = ot.unif(C2.shape[0])
    dist = ot.gromov_wasserstein2(C1, C2, p, q, "square_loss")
    return dist


def gromov_dtw_dist(GDTW: gromov_dtw, x1, x2):
    x1 = torch.from_numpy(x1)
    x2 = torch.from_numpy(x2)
    result = GDTW.forward(x1, x2)
    return result.item()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size", type=float, default=0.5)
    parser.add_argument("--outlier_ratio", type=float)
    parser.add_argument("--random_outlier", action="store_true")
    parser.add_argument("--metric", type=str, default="euclidean")
    parser.add_argument("--m", type=float, default=None)
    parser.add_argument("--reg", type=float, default=10)
    parser.add_argument("--without_outlier", action="store_true")
    parser.add_argument(
        "--algo",
        type=str,
        choices=["parial-order-gromov", "gromov-dtw", "gromov", "partial-gromov"],
    )
    args = parser.parse_args()
    return args


def main(args):
    logger.info(f"Args: {args}")
    logger.info(f"Using algo: {args.algo}")
    if args.algo == "gromov-dtw":
        GDTW = gromov_dtw(
            max_iter=10, gamma=0.1, loss_only=1, dtw_approach="soft_GDTW", verbose=0
        )

    # convert args to dict
    evaluate = vars(args)

    # Dataset
    weis_dataset = WeisDataset.from_folder(
        MULTI_WEI_PATH, test_size=0.5, multi_feature=True
    )
    X_train = [weis_dataset.get_sequence(idx) for idx in weis_dataset.train_idx]
    X_test = [weis_dataset.get_sequence(idx) for idx in weis_dataset.test_idx]
    y_train = (weis_dataset.get_label(idx) for idx in weis_dataset.train_idx)
    y_test = (weis_dataset.get_label(idx) for idx in weis_dataset.test_idx)
    y_train = np.array(list(y_train))
    y_test = np.array(list(y_test))

    # Add outlier
    if args.without_outlier:
        logger.info("Without outlier")
    elif args.random_outlier:
        logger.info("random outlier is True then outlier ratio is ignored")
        X_train = list(map(lambda x: add_outlier(x, random=True), X_train))
        X_test = list(map(lambda x: add_outlier(x, random=True), X_test))
    elif args.outlier_ratio is None:
        raise ValueError("Outlier ratio must be specified")
    else:
        logger.info(f"Add outlier with ratio: {args.outlier_ratio}")
        X_train = list(
            map(lambda x: add_outlier(x, outlier_ratio=args.outlier_ratio), X_train)
        )
        X_test = list(
            map(lambda x: add_outlier(x, outlier_ratio=args.outlier_ratio), X_test)
        )

    train_size = len(X_train)
    test_size = len(X_test)

    logger.info(f"Train size: {train_size}, Test size: {test_size}")

    result = np.zeros((test_size, train_size))
    for train_idx in tqdm(range(train_size)):
        for test_idx in tqdm(range(test_size), leave=False):
            if args.algo == "parial-order-gromov":
                distance = partial_order_gromov_dist(
                    X_train[train_idx],
                    X_test[test_idx],
                    metric=args.metric,
                    order_reg=args.reg,
                    m=args.m,
                )
            elif args.algo == "gromov-dtw":
                distance = gromov_dtw_dist(
                    GDTW,
                    X_train[train_idx],
                    X_test[test_idx],
                )
            elif args.algo == "gromov":
                distance = gromov_dist(
                    X_train[train_idx],
                    X_test[test_idx],
                    metric=args.metric,
                )
            elif args.algo == "partial-gromov":
                distance = partial_gromov_dist(
                    X_train[train_idx],
                    X_test[test_idx],
                    metric=args.metric,
                    m=args.m,
                )
            else:
                raise ValueError(f"Unknown algo: {args.algo}")
            result[test_idx, train_idx] = distance

    for k in [1, 3, 5]:
        y_pred = knn_classifier_from_distance_matrix(
            distance_matrix=result,
            k=k,
            labels=y_train,
        )
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_micro = f1_score(y_test, y_pred, average="micro")
        logger.info(f"Accuracy with k = {k}: {accuracy}")
        logger.info(f"F1 macro with k = {k}: {f1_macro}")
        logger.info(f"F1 micro with k = {k}: {f1_micro}")
        evaluate[f"accuracy_{k}"] = accuracy
        evaluate[f"f1_macro_{k}"] = f1_macro
        evaluate[f"f1_micro_{k}"] = f1_micro

    logger.info("-" * 50)
    # write evaluate to file
    folder_path = os.path.dirname(os.path.abspath(__file__))
    evaluate_path = os.path.join(folder_path, "result.txt")
    with open(evaluate_path, "a") as f:
        f.write(str(evaluate) + "\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)

    # get folder path of this file
    folder_path = os.path.dirname(os.path.abspath(__file__))
