import numpy as np


def add_outlier(x, outlier_ratio=None, random=False):
    # x have shape (time, feature)
    #   random add outlier to x at random time, outlier is a random 0.5 0,1 have shape like feature
    def get_outlier(x):
        # import ipdb; ipdb.set_trace()
        outlier = np.random.randint(np.min(x), np.max(x), size=x.shape[1])
        return outlier

    x = x.copy()
    if random:
        # print("outlier_ratio is ignored")
        outlier_ratio = np.random.uniform(0, 0.2)

    outlier_idx = np.random.choice(
        range(x.shape[0]), size=int(x.shape[0] * outlier_ratio), replace=False
    )
    for idx in outlier_idx:
        x[idx] = get_outlier(x)
    return x
