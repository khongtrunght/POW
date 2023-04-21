import torch
import numpy as np
from src.pow.pow import pow_regularization, pow_dst_matrix_and_margin
import pytest


def POT_feature_2sides(a,b,D, m=None, nb_dummies=1):
  # a = np.ones(D.shape[0])/D.shape[0]
  # b = np.ones(D.shape[1])/D.shape[1]
  if m < 0:
      raise ValueError("Problem infeasible. Parameter m should be greater"
                        " than 0.")
  elif m > np.min((np.sum(a), np.sum(b))):
      raise ValueError("Problem infeasible. Parameter m should lower or"
                        " equal than min(|a|_1, |b|_1).")
  b_extended = np.append(b, [(np.sum(a) - m) / nb_dummies] * nb_dummies)
  a_extended = np.append(a, [(np.sum(b) - m) / nb_dummies] * nb_dummies)
  D_extended = np.zeros((len(a_extended), len(b_extended)))
  D_extended[-nb_dummies:, -nb_dummies:] = np.max(D) * 2
  D_extended[:len(a), :len(b)] = D
  return a_extended, b_extended, D_extended

def POT_feature_1side(a,b,D, m=0.7, nb_dummies=1):
  # a = np.ones(D.shape[0])*m/D.shape[0]
  # b = np.ones(D.shape[1])/D.shape[1]
  a = a*m
  '''drop on side b --> and dummpy point on side a'''
  a_extended = np.append(a, [(np.sum(b) - m) / nb_dummies] * nb_dummies)
  D_extended = torch.zeros((len(a_extended), len(b)))
#   D_extended = F.pad(input=D, pad=(0, 0, 0, 1), mode='constant', value=0)
  D_extended[:len(a), :len(b)] = D
  return a_extended, b,D_extended

def order_regularization(M,reg):
        I = np.zeros_like(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                I[i, j] = (i/M.shape[0] - j/M.shape[1])**2
        return M + reg*I

def compute_OPW_costs(D ,reg, m=None, dropBothSides = False):
    a = np.ones(D.shape[0])/D.shape[0]
    b = np.ones(D.shape[1])/D.shape[1]

    D = order_regularization(D, reg)
    if dropBothSides:
        a,b,D = POT_feature_2sides(a,b,D,m)
    else:
        #drop side b
        a,b,D = POT_feature_1side(a,b,D,m)

    a = torch.from_numpy(a).to(D.device)
    b = torch.from_numpy(b).to(D.device)
    return D,a,b



def test_pow_regularization_new(M, reg):

    new_distance = pow_regularization(M, reg).detach().numpy()
    old_distance = order_regularization(M, reg)
    assert np.allclose(new_distance, old_distance)


def test_pow_cost(M, reg, m):
    M_old, a_old, b_old = compute_OPW_costs(M, reg, m)
    M_new, a_new, b_new = pow_dst_matrix_and_margin(M, reg, m)

    assert np.allclose(M_old, M_new.detach().numpy())
    assert np.allclose(a_old, a_new.detach().numpy())
    assert np.allclose(b_old, b_new.detach().numpy())
