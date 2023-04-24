import numpy as np
import ot
import torch


def pow_regularization(M, reg):
    rows, cols = M.shape
    i, j = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
    i, j = i.to(M.device), j.to(M.device)
    I = ((i / rows - j / cols) ** 2).to(M.dtype).to(M.device)
    return M + reg * I


def partial_extend_2side(a, b, M, m=None, nb_dummies=1):
    if m > torch.min(torch.sum(a), torch.sum(b)):
        raise ValueError(
            "Problem infeasible. Parameter m should lower or"
            " equal than min(|a|_1, |b|_1)."
        )
    # import ipdb; ipdb.set_trace()
    b_extended = torch.cat(
        (b, torch.ones(nb_dummies) * (torch.sum(a) - m) / nb_dummies), dim=0
    )
    a_extended = torch.cat(
        (a, torch.ones(nb_dummies) * (torch.sum(b) - m) / nb_dummies), dim=0
    )
    D_extended = torch.zeros((len(a_extended), len(b_extended)), dtype=M.dtype)
    D_extended[-nb_dummies:, -nb_dummies:] = torch.max(M) * 2
    D_extended[: len(a), : len(b)] = M
    return a_extended, b_extended, D_extended


def partial_extend(normal_side, drop_side, M, m=None, nb_dummies=1):
    """Extend cost matrix to include dummy nodes

    Args:
        a (torch.tensor): normal side
        b (torch.tensor): drop side
        M (torch.tensor): cost matrix
        m (float, optional): keep rate. Defaults to None.
    """
    normal_side = normal_side * m
    a_extended = torch.cat(
        (
            normal_side,
            torch.ones(nb_dummies) * ((torch.sum(drop_side) - m) / nb_dummies),
        ),
        dim=0,
    )
    D_extended = torch.zeros((len(a_extended), len(drop_side)), dtype=M.dtype)
    D_extended[: len(normal_side), : len(drop_side)] = M
    return a_extended, drop_side, D_extended


def pow_dst_matrix_and_margin(M, reg, m, extend="1side"):
    rows, cols = M.shape
    if type(M) == np.ndarray:
        M = torch.from_numpy(M)
    a = torch.ones(rows, dtype=M.dtype, device=M.device) / rows
    b = torch.ones(cols, dtype=M.dtype, device=M.device) / cols

    M = pow_regularization(M, reg)
    if extend == "1side":
        a, b, M = partial_extend(normal_side=a, drop_side=b, M=M, m=m, nb_dummies=1)
    elif extend == "2side":
        a, b, M = partial_extend_2side(a=a, b=b, M=M, m=m, nb_dummies=1)
    return M, a, b


def get_assignment(soft_assignment):
    """Get assignment from soft assignment"""
    if torch.is_tensor(soft_assignment):
        soft_assignment = soft_assignment.detach().cpu().numpy()
    assignment = np.argmax(soft_assignment, axis=0)
    outlier_label = soft_assignment.shape[0] - 1
    assignment[assignment == outlier_label] = -1
    return assignment


def pow_distance(M, reg, m, extend="1side"):
    M, a, b = pow_dst_matrix_and_margin(M, reg, m, extend)
    # return ot.sinkhorn2(a, b, M, reg=0.01) #FIXME: hardcoded reg, numerical error
    return ot.emd2(a, b, M)
