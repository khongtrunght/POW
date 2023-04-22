import numpy as np
import ot
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from config.config import logger
from src.baselines.opw import opw_discrepancy
from src.baselines.softdtw import soft_dtw_discrepancy
from src.baselines.topw1 import t_opw1
from src.dp.exact_dp import drop_dtw_distance, dtw_distance
from src.pow.pow import pow_distance


def knn_classifier_from_distance_matrix(distance_matrix, k, labels):
    """
    Computes the k-nearest neighbors for each point in a dataset given a precomputed distance matrix
    and returns the predicted class labels based on the majority class of its k-nearest neighbors.

    Parameters:
    -----------
    distance_matrix: array-like or sparse matrix, shape (n_test_samples, n_train_samples)
        The precomputed distance matrix.
    k: int
        The number of neighbors to use for classification.
    labels: array-like, shape (n_train_samples,)
        The class labels for each data point in the dataset.

    Returns:
    --------
    predicted_labels: array-like, shape (n_samples,)
        The predicted class labels for each point in the dataset.
    """
    if np.min(distance_matrix) < 0:
        distance_matrix = distance_matrix - np.min(distance_matrix)
    knn_clf = KNeighborsClassifier(
        n_neighbors=k, algorithm="brute", metric="precomputed"
    )
    n_train_samples = distance_matrix.shape[1]
    knn_clf.fit(np.random.rand(n_train_samples, n_train_samples), labels)
    predicted_labels = knn_clf.predict(distance_matrix)
    return predicted_labels


fn_dict = {
    "pow": pow_distance,
    "dtw": dtw_distance,
    "drop_dtw": drop_dtw_distance,
    "topw1": t_opw1,
    "opw": opw_discrepancy,
    "softdtw": soft_dtw_discrepancy,
}


def get_distance_matrix(X_train, X_test, args):
    train_size = len(X_train)
    test_size = len(X_test)
    logger.info(f"Train size: {train_size}")
    logger.info(f"Test size: {test_size}")

    result = np.zeros((test_size, train_size))
    for train_idx in tqdm(range(train_size)):
        for test_idx in tqdm(range(test_size), leave=False):
            x_tr = X_train[train_idx].reshape(-1, 1)
            x_te = X_test[test_idx].reshape(-1, 1)
            M = ot.dist(x_tr, x_te, metric=args.distance)
            M = M / M.max()
            if args.metric == "pow":
                distance = fn_dict[args.metric](M, m=args.m, reg=args.reg)
            elif args.metric == "drop_dtw":
                distance = fn_dict[args.metric](M, keep_percentile=args.m)
            elif args.metric == "topw1":
                distance = fn_dict[args.metric](
                    X=X_train[train_idx], Y=X_test[test_idx], metric=args.distance
                )
            else:
                distance = fn_dict[args.metric](M)
            # elif args.metric == "dtw":
            #     distance = fn_dict[args.metric](M)

            # elif args.metric == "pow":
            #     distance = fn_dict[args.metric](M)
            # elif args.metric == "softdtw":
            #     distance = fn_dict[args.metric](M)
            if (distance == np.inf) or np.isnan(distance):
                distance = np.max(result)
            result[test_idx, train_idx] = distance
    return result
