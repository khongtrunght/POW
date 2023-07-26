import ot
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.pogw.pogw_2 import partial_order_gromov_wasserstein
from src.gdtw.GDTW import gromov_dtw
from src.experiments.digit_moving.utils import add_outlier, random_swap, get_ground_truth, soft_assigment_to_matching
from src.experiments.digit_moving.utils import accuracy, iou
from ot.gromov import gromov_wasserstein
from ot.partial import partial_gromov_wasserstein
import torch
from glob import glob

def GDTW_alignment(x1, x2):
    GDTW = gromov_dtw(
            max_iter=5, gamma=0.1, loss_only=0, dtw_approach="GDTW", verbose=0
        )
    x1 = torch.from_numpy(x1)
    x2 = torch.from_numpy(x2)
    result = GDTW.forward(x1, x2)[1].numpy()
    predicted_matching = soft_assigment_to_matching(result)
    return predicted_matching

def POGW_alignment(X, Y):
    C1 = ot.dist(X,metric="euclidean").astype(np.float64)
    C2 = ot.dist(Y,metric="euclidean").astype(np.float64)

    C1 = C1 / C1.mean()
    C2 = C2 / C2.mean()

    p1 = ot.unif(C1.shape[0])
    p2 = ot.unif(C2.shape[0])

    T = partial_order_gromov_wasserstein(C1,C2,p1,p2,m = 20/22, order_reg=0.003, return_dist=False)

    predicted_matching = soft_assigment_to_matching(T)
    return predicted_matching

def GW_alignment(X, Y):
    C1 = ot.dist(X,metric="euclidean").astype(np.float64)
    C2 = ot.dist(Y,metric="euclidean").astype(np.float64)

    C1 = C1 / C1.mean()
    C2 = C2 / C2.mean()

    p = ot.unif(C1.shape[0])
    q = ot.unif(C2.shape[0])

    T = gromov_wasserstein(C1, C2, p, q)

    predicted_matching = soft_assigment_to_matching(T)
    return predicted_matching

def PGW_alignment(X, Y):
    C1 = ot.dist(X,metric="euclidean").astype(np.float64)
    C2 = ot.dist(Y,metric="euclidean").astype(np.float64)

    C1 = C1 / C1.mean()
    C2 = C2 / C2.mean()

    p = ot.unif(C1.shape[0])
    q = ot.unif(C2.shape[0])

    T = partial_gromov_wasserstein(C1, C2, p, q, m = None)

    predicted_matching = soft_assigment_to_matching(T)
    return predicted_matching


def main():
    lst_acc = []
    lst_iou = []
    print("Total pairs: ", len(glob("Datasets/mnist_moving/*")))
    for x in range(1,51):
        for digit in range(10):
            mnist_data = np.load(f"Datasets/mnist_moving/{x}/{digit}.npy").astype(np.float64)
            usps_data = np.load(f"Datasets/usps_moving/{x}/{digit}.npy").astype(np.float64)
            mnist_order = [i for i in range(mnist_data.shape[0])]
            usps_order = [i for i in range(usps_data.shape[0])]
            # print(mnist_data.shape, usps_data.shape)
            # print(mnist_data.max(), mnist_data.min())
            # print(usps_data.max(), usps_data.min())


            mnist_data, mnist_order = random_swap(mnist_data, mnist_order)
            mnist_data, mnist_order = add_outlier(mnist_data, mnist_order, n_outliers=2, norm=False)
            usps_data, usps_order = add_outlier(usps_data, usps_order, n_outliers=2, norm=False)


            flatten_usps_data = np.array([i.flatten() for i in usps_data])
            flatten_mnist_data = np.array([i.flatten() for i in mnist_data])


            ground_truth_matching = get_ground_truth(mnist_order, usps_order)


            predicted_matching = POGW_alignment(flatten_mnist_data, flatten_usps_data)
            # print("Accuracy: ", accuracy(ground_truth_matching, predicted_matching))
            # print("IoU: ", iou(ground_truth_matching, predicted_matching))
            lst_acc.append(accuracy(ground_truth_matching, predicted_matching))
            lst_iou.append(iou(ground_truth_matching, predicted_matching))

    print(len(lst_acc))
    print("Accuracy: ", sum(lst_acc)/len(lst_acc))
    print("IoU: ", sum(lst_iou)/len(lst_iou))
main()