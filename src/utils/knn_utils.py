import numpy as np
from sklearn.neighbors import KNeighborsClassifier


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
    knn_clf = KNeighborsClassifier(
        n_neighbors=k, algorithm="brute", metric="precomputed"
    )
    n_train_samples = distance_matrix.shape[1]
    knn_clf.fit(np.random.rand(n_train_samples, n_train_samples), labels)
    predicted_labels = knn_clf.predict(distance_matrix)
    return predicted_labels
