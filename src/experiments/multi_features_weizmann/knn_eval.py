import argparse

import numpy as np
import ot
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from config.config import MULTI_WEI_PATH, logger
from src.experiments.multi_features_weizmann.dataset import WeisDataset
from src.experiments.weizmann.utils import add_outlier
from src.pogw.pogw import partial_gromov_wasserstein
from src.utils.knn_utils import knn_classifier_from_distance_matrix

# np.random.seed(42)
# random.seed(42)
# sklearn_seed = 0


def order_gromov_dist(x1, x2, m=None, metric="cosine", order_reg=0.1):
    C1 = ot.dist(x1, x1, metric=metric)
    C2 = ot.dist(x2, x2, metric=metric)
    C1 = C1 / C1.max()
    C2 = C2 / C2.max()
    p = ot.unif(C1.shape[0])
    q = ot.unif(C2.shape[0])

    dist = partial_gromov_wasserstein(
        C1, C2, p, q, m=m, order_reg=order_reg, return_dist=True
    )

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size", type=float, default=0.5)
    parser.add_argument("--outlier_ratio", type=float, default=0.1)
    parser.add_argument("--metric", type=str, default="eucledian")
    parser.add_argument("--m", type=float, default=None)
    parser.add_argument("--reg", type=int, default=10)
    parser.add_argument("--k", type=int, default=1)
    args = parser.parse_args()
    return args


def main(args):
    logger.info(f"Args: {args}")
    weis_dataset = WeisDataset.from_folder(
        MULTI_WEI_PATH, test_size=0.5, multi_feature=True
    )
    X_train = [weis_dataset.get_sequence(idx) for idx in weis_dataset.train_idx]
    X_test = [weis_dataset.get_sequence(idx) for idx in weis_dataset.test_idx]
    X_train = list(
        map(lambda x: add_outlier(x, outlier_ratio=args.outlier_ratio), X_train)
    )
    X_test = list(
        map(lambda x: add_outlier(x, outlier_ratio=args.outlier_ratio), X_test)
    )

    y_train = (weis_dataset.get_label(idx) for idx in weis_dataset.train_idx)
    y_test = (weis_dataset.get_label(idx) for idx in weis_dataset.test_idx)
    y_train = np.array(list(y_train))
    y_test = np.array(list(y_test))

    train_size = len(X_train)
    test_size = len(X_test)

    logger.info(f"Train size: {train_size}")
    logger.info(f"Test size: {test_size}")

    result = np.zeros((test_size, train_size))
    for train_idx in tqdm(range(train_size)):
        for test_idx in tqdm(range(test_size), leave=False):
            distance = order_gromov_dist(
                X_train[train_idx],
                X_test[test_idx],
                metric=args.metric,
                order_reg=args.reg,
                m=args.m,
            )
            # distance = gromov_dist(
            #     X_train[train_idx],
            #     X_test[test_idx],
            #     metric=args.metric,
            # )
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
