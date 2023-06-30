import numpy as np

def sample_outlier(shape, norm=True):
    if norm:
        outlier = np.random.uniform(0, 1, shape)
    else:
        outlier = np.random.randint(0, 255, shape)
    return outlier

def add_outlier(X,order, n_outliers=1, norm=True):
    # Insert random outliers
    idx = np.random.randint(0, X.shape[0], n_outliers)
    for i in idx:
        outlier_shape = (1,*X[i].shape)
        X = np.vstack((X[:i], sample_outlier(outlier_shape, norm),X[i:]))
        order.insert(i, -1)
    return X, order

def random_swap(X, order):
    # sample n_swaps indices
    idx = np.random.randint(0, X.shape[0]-1)
    # swap
    tmp = X[idx].copy()
    X[idx] = X[idx+1].copy()
    X[idx+1] = tmp
    # update order
    order[idx], order[idx+1] = order[idx+1], order[idx]
    return X, order

def get_ground_truth(X_order, Y_order):
    ground_truth_matrix = np.zeros((len(X_order), len(Y_order)))
    matching = []
    for i in range(len(X_order)):
        for j in range(len(Y_order)):
            if X_order[i] == -1 or Y_order[j] == -1:
                continue
            if X_order[i] == Y_order[j]:
                ground_truth_matrix[i,j] = 1
                matching.append((i,j))

    for i in range(len(X_order)):
        if X_order[i] == -1:
            matching.append((i,-1))
    for j in range(len(Y_order)):
        if Y_order[j] == -1:
            matching.append((-1,j))

    matching = sorted(matching, key=lambda x: x[0])
    return matching

def get_ground_truth_matrix(X_order, Y_order):
    ground_truth_matrix = np.zeros((len(X_order), len(Y_order)))
    for i in range(len(X_order)):
        for j in range(len(Y_order)):
            if X_order[i] == -1 or Y_order[j] == -1:
                continue
            if X_order[i] == Y_order[j]:
                ground_truth_matrix[i,j] = 1
    return ground_truth_matrix

def soft_assigment_to_matching(T):
    T = T.copy().round(4)
    matching = []
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            if T[i,j] > 1e-2:
                matching.append((i,j))

    for i in range(T.shape[0]):
        if T[i].sum() < 1e-3:
            matching.append((i,-1))
    for j in range(T.shape[1]):
        if T[:,j].sum() < 1e-3:
            matching.append((-1,j))
    matching = sorted(matching, key=lambda x: x[0])
    return matching


def accuracy(predicted_matching, ground_truth_matching):
    correct = 0
    for i in predicted_matching:
        if i in ground_truth_matching:
            correct += 1
    return correct/len(predicted_matching)

def iou(predicted_matching, ground_truth_matching):
    correct = 0
    for i in predicted_matching:
        if i in ground_truth_matching:
            correct += 1
    return correct/(len(predicted_matching) + len(ground_truth_matching) - correct)