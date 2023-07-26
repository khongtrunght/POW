import numpy as np
import ot
from ot.lp import emd

def order_regularization(M, reg):
    I = np.zeros_like(M)
    rows, cols = M.shape
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            I[i, j] = np.abs(i / rows - j / cols)**2

    return M + reg * I

def gwgrad_partial(C1, C2, T):
    """Compute the GW gradient. Note: we can not use the trick in :ref:`[12] <references-gwgrad-partial>`
    as the marginals may not sum to 1.

    Parameters
    ----------
    C1: array of shape (n_p,n_p)
        intra-source (P) cost matrix

    C2: array of shape (n_u,n_u)
        intra-target (U) cost matrix

    T : array of shape(n_p+nb_dummies, n_u) (default: None)
        Transport matrix

    Returns
    -------
    numpy.array of shape (n_p+nb_dummies, n_u)
        gradient


    .. _references-gwgrad-partial:
    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """
    cC1 = np.dot(C1**2 / 2, np.dot(T, np.ones(C2.shape[0]).reshape(-1, 1)))
    cC2 = np.dot(np.dot(np.ones(C1.shape[0]).reshape(1, -1), T), C2**2 / 2)
    constC = cC1 + cC2
    A = -np.dot(C1, T).dot(C2.T)
    tens = constC + A
    return tens * 2

def gwgrad_partial_2(C1, C2, T, order_reg):
    """Compute the GW gradient. Note: we can not use the trick in :ref:`[12] <references-gwgrad-partial>`
    as the marginals may not sum to 1.

    Parameters
    ----------
    C1: array of shape (n_p,n_p)
        intra-source (P) cost matrix

    C2: array of shape (n_u,n_u)
        intra-target (U) cost matrix

    T : array of shape(n_p+nb_dummies, n_u) (default: None)
        Transport matrix

    Returns
    -------
    numpy.array of shape (n_p+nb_dummies, n_u)
        gradient


    .. _references-gwgrad-partial:
    References
    ----------
    .. [12] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """
    M = np.zeros((C1.shape[0], C1.shape[0], C2.shape[0], C2.shape[0]))
    # print(M.shape)
    for i in range(M.shape[0]):
        for k in range(M.shape[1]):
            for j in range(M.shape[2]):
                for l in range(M.shape[3]):
                    M[i,k,j,l] = np.abs(C1[i,k] - C2[j,l])**2 + order_reg*np.abs(i/T.shape[0] - j/T.shape[1])
    
    # gw_grad = np.zeros_like(T)
    # for i in range(T.shape[0]):
    #     for j in range(T.shape[1]):
    #         gw_grad[i,j] = np.sum(M[i,:,j,:]*T[i,j])

    gw_grad = np.einsum('ijkl, kl -> ij', M, T)
    # print(gw_grad.shape)
    return gw_grad


def gwgrad_partial_3(C1, C2, T):
    diff_squared = np.abs(C1[:, None, :, None] - C2[None, :, None, :])**2

    # Perform the einsum operation to calculate gw_grad
    gw_grad = np.einsum('ijkl,kl->ij', diff_squared, T)

    return gw_grad


def gwgrad_partial_4(C1, C2, T, order_reg):
    diff_squared = np.abs(C1[:, None, :, None] - C2[None, :, None, :])**2

    i = np.arange(C1.shape[0]).reshape(-1, 1)
    j = np.arange(C2.shape[0]).reshape(1, -1)
    diff = (i/i.shape[0] - j/j.shape[1])**2
    M_2 = np.zeros((C1.shape[0], C2.shape[0],C1.shape[0], C2.shape[0]))
    for k in range(M_2.shape[2]):
        for l in range(M_2.shape[3]):
            M_2[:,:,k,l] = diff
    M = diff_squared + order_reg*M_2
    # Perform the einsum operation to calculate gw_grad
    gw_grad = np.einsum('ijkl,kl->ij', M, T)

    return gw_grad



def gwloss_partial(C1, C2, T, order_reg):
    """Compute the GW loss.

    Parameters
    ----------
    C1: array of shape (n_p,n_p)
        intra-source (P) cost matrix

    C2: array of shape (n_u,n_u)
        intra-target (U) cost matrix

    T : array of shape(n_p+nb_dummies, n_u) (default: None)
        Transport matrix

    Returns
    -------
    GW loss
    """
    # g = gwgrad_partial(C1, C2, T) * 0.5
    g = gwgrad_partial_4(C1, C2, T, order_reg)
    # g = gwgrad_partial_3(C1, C2, T) + np.sum(order_regularization(T, 0.003))
    return np.sum(g * T)


def partial_order_gromov_wasserstein(
    C1,
    C2,
    p,
    q,
    m,
    order_reg,
    nb_dummies=1,
    G0=None,
    thres=1,
    numItermax=1000,
    tol=1e-7,
    log=False,
    verbose=False,
    return_dist=False,
    ot_algo="emd",
    sinkhorn_reg=0.1,
    **kwargs
):
    r"""
    Solves the partial optimal transport problem
    and returns the OT plan

    The function considers the following problem:

    .. math::
        \gamma = \mathop{\arg \min}_\gamma \quad \langle \gamma, \mathbf{M} \rangle_F

    .. math::
        s.t. \ \gamma \mathbf{1} &\leq \mathbf{a}

             \gamma^T \mathbf{1} &\leq \mathbf{b}

             \gamma &\geq 0

             \mathbf{1}^T \gamma^T \mathbf{1} = m &\leq \min\{\|\mathbf{a}\|_1, \|\mathbf{b}\|_1\}

    where :

    - :math:`\mathbf{M}` is the metric cost matrix
    - :math:`\Omega` is the entropic regularization term, :math:`\Omega(\gamma) = \sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - :math:`\mathbf{a}` and :math:`\mathbf{b}` are the sample weights
    - `m` is the amount of mass to be transported

    The formulation of the problem has been proposed in
    :ref:`[29] <references-partial-gromov-wasserstein>`


    Parameters
    ----------
    C1 : ndarray, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : ndarray, shape (nt, nt)
        Metric costfr matrix in the target space
    p : ndarray, shape (ns,)
        Distribution in the source space
    q : ndarray, shape (nt,)
        Distribution in the target space
    m : float, optional
        Amount of mass to be transported
        (default: :math:`\min\{\|\mathbf{p}\|_1, \|\mathbf{q}\|_1\}`)
    nb_dummies : int, optional
        Number of dummy points to add (avoid instabilities in the EMD solver)
    G0 : ndarray, shape (ns, nt), optional
        Initialisation of the transportation matrix
    thres : float, optional
        quantile of the gradient matrix to populate the cost matrix when 0
        (default: 1)
    numItermax : int, optional
        Max number of iterations
    tol : float, optional
        tolerance for stopping iterations
    log : bool, optional
        return log if True
    verbose : bool, optional
        Print information along iterations
    **kwargs : dict
        parameters can be directly passed to the emd solver


    Returns
    -------
    gamma : (dim_a, dim_b) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary returned only if `log` is `True`


    Examples
    --------
    >>> import ot
    >>> import scipy as sp
    >>> a = np.array([0.25] * 4)
    >>> b = np.array([0.25] * 4)
    >>> x = np.array([1,2,100,200]).reshape((-1,1))
    >>> y = np.array([3,2,98,199]).reshape((-1,1))
    >>> C1 = sp.spatial.distance.cdist(x, x)
    >>> C2 = sp.spatial.distance.cdist(y, y)
    >>> np.round(partial_gromov_wasserstein(C1, C2, a, b),2)
    array([[0.  , 0.25, 0.  , 0.  ],
           [0.25, 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.25, 0.  ],
           [0.  , 0.  , 0.  , 0.25]])
    >>> np.round(partial_gromov_wasserstein(C1, C2, a, b, m=0.25),2)
    array([[0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.  , 0.  ],
           [0.  , 0.  , 0.25, 0.  ],
           [0.  , 0.  , 0.  , 0.  ]])


    .. _references-partial-gromov-wasserstein:
    References
    ----------
    ..  [29] Chapel, L., Alaya, M., Gasso, G. (2020). "Partial Optimal
        Transport with Applications on Positive-Unlabeled Learning".
        NeurIPS.

    """

    if m is None:
        m = np.min((np.sum(p), np.sum(q)))
    elif m < 0:
        raise ValueError("Problem infeasible. Parameter m should be greater" " than 0.")
    elif m > np.min((np.sum(p), np.sum(q))):
        raise ValueError(
            "Problem infeasible. Parameter m should lower or"
            " equal than min(|a|_1, |b|_1)."
        )

    if G0 is None:
        G0 = np.outer(p, q)

    dim_G_extended = (len(p) + nb_dummies, len(q) + nb_dummies)
    q_extended = np.append(q, [(np.sum(p) - m) / nb_dummies] * nb_dummies)
    p_extended = np.append(p, [(np.sum(q) - m) / nb_dummies] * nb_dummies)

    cpt = 0
    err = 1

    if log:
        log = {"err": []}

    while err > tol and cpt < numItermax:

        Gprev = np.copy(G0)

        # M = gwgrad_partial_2(C1, C2, G0, order_reg)
        # M = gwgrad_partial_3(C1, C2, G0)
        M = gwgrad_partial_4(C1, C2, G0, order_reg)
        # M = gwgrad_partial(C1, C2, G0)
        # -----------------------
        # M = order_regularization(M, order_reg)
        # -----------------------
        M_emd = np.zeros(dim_G_extended)
        M_emd[: len(p), : len(q)] = M
        M_emd[-nb_dummies:, -nb_dummies:] = 1e2
        M_emd = np.asarray(M_emd, dtype=np.float64)
        if ot_algo == "emd":
            Gc, logemd = emd(p_extended, q_extended, M_emd, log=True, **kwargs)
        elif ot_algo == "sinkhorn":
            Gc = ot.sinkhorn(p_extended, q_extended, M_emd, reg=sinkhorn_reg, log=False)

        # -----------------------
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # plt.figure(figsize=(10,10))
        # sns.heatmap(M_emd.round(4),cmap="YlGnBu",annot = True)
        # sns.heatmap(Gc.round(4),cmap="YlGnBu",annot = True)
        # plt.show()
        # -----------------------
        G0 = Gc[: len(p), : len(q)]

        if cpt % 10 == 0:  # to speed up the computations
            err = np.linalg.norm(G0 - Gprev)
            if log:
                log["err"].append(err)
            if verbose:
                if cpt % 200 == 0:
                    print(
                        "{:5s}|{:12s}|{:12s}".format("It.", "Err", "Loss")
                        + "\n"
                        + "-" * 31
                    )
                print("{:5d}|{:8e}|{:8e}".format(cpt, err, gwloss_partial(C1, C2, G0, order_reg)))

        deltaG = G0 - Gprev
        a = gwloss_partial(C1, C2, deltaG, order_reg)
        b = 2 * np.sum(M * deltaG)
        if b > 0:  # due to numerical precision
            gamma = 0
            cpt = numItermax
        elif a > 0:
            gamma = min(1, np.divide(-b, 2.0 * a))
        else:
            if (a + b) < 0:
                gamma = 1
            else:
                gamma = 0
                cpt = numItermax

        G0 = Gprev + gamma * deltaG
        cpt += 1

    if return_dist:
        return gwloss_partial(C1, C2, G0, order_reg)
    else:
        return G0[: len(p), : len(q)]
        # return G0
