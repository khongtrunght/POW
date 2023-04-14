import numpy as np


def add_outlier(x, outlier_ratio=0.1):
    # x have shape (time, feature)
    #   random add outlier to x at random time, outlier is a random 0.5 0,1 have shape like feature
    x = x.copy()

    # import ipdb; ipdb.set_trace()
    def get_outlier(x):
        outlier = np.random.randint(0, 2, size=x.shape[1])
        return outlier

    outlier_idx = np.random.choice(
        range(x.shape[0]), size=int(x.shape[0] * outlier_ratio), replace=False
    )
    for idx in outlier_idx:
        x[idx] = get_outlier(x)
    return x
