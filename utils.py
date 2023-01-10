import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from numpy.linalg import solve

inf = np.inf


# Occupancy measure
def d_pi(gamma, P_pi, d0):
    nS, _ = P_pi.shape
    dpi = solve((np.eye(nS) - gamma * P_pi).T, d0) * (1 - gamma)
    return dpi

# V = maxQ(s,a)
def QtoV(Q):
    return np.max(Q, axis=1)


def VtoQ(V, P, R, gamma):
    nS, nA = R.shape
    # return R + gamma * np.inner(P.reshape(nS, nA, nS), V)
    return R + gamma * np.max(np.inner(P.reshape(nS, nA, nS), V), axis=1)


def QtoPolicy(Q):
    nS, nA = Q.shape
    pi = np.zeros((nS, nA))
    pi[np.arange(nS), np.argmax(Q, axis=1)] = 1
    return pi


def evaluate_MEObj_from_policy(pi, R, Ps, d_0, gamma):
    nS, nA = R.shape
    R_pi = np.sum(pi * R, axis=1)
    pi_axis = pi[:, :, np.newaxis]
    Ps_pi = [np.sum(P.reshape(nS, nA, nS) * pi_axis, axis=1) for P in Ps]
    Vs = [solve(np.eye(nS) - gamma * P_pi, R_pi) for P_pi in Ps_pi]
    avg_V = np.array([np.inner(V, d_0) for V in Vs])
    ME_Obj = np.mean(avg_V)
    return ME_Obj


def value_iteration(V, P, R, gamma):
    nS, nA = R.shape
    Q = VtoQ(V, P, R, gamma)
    # Return the new V and the new deterministic policy
    # 以下是修改部分
    #new_V = np.max(Q, axis=1)
    new_pi_idx = np.argmax(Q, axis=1)
    new_pi = (np.arange(nA) == new_pi_idx[:, np.newaxis]).astype(float)
    return Q, new_pi


# epsilon represent the scale of perturbation
# Orth
def generate_mix_Ps_Orth(n, nS, nA, eps_lst=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), seed=250):
    np.random.seed(seed)
    eps_lst_len = len(eps_lst)
    mix_Ps = []

    sparsity = 0.5
    sparse = np.random.binomial(1, sparsity, size=(nA * nS, nS))
    for i in range(nA * nS):
        if sum(sparse[i, :]) == 0:
            sparse[i, np.random.randint(nS)] = 1
        if sum(sparse[i, :]) == nS:
            sparse[i, np.random.randint(nS)] = 0

    inverse_sparse = 1 - sparse

    center_P = np.random.uniform(0, 1, size=(nA * nS, nS)) * sparse

    center_P = center_P / np.sum(center_P, 1)[:, np.newaxis]
    noncenter_Ps = []
    for _ in range(n - 1):
        U = np.random.uniform(0, 1, size=(nA * nS, nS)) * inverse_sparse
        U = U / np.sum(U, 1)[:, np.newaxis]
        noncenter_Ps.append(U.copy())
    for i in range(eps_lst_len):
        eps = eps_lst[i]
        current_Ps = []
        current_Ps.append(center_P.copy())
        for j in range(n - 1):
            current_P = center_P * (1 - eps) + eps * noncenter_Ps[j]
            current_Ps.append(current_P.copy())
        mix_Ps.append(current_Ps)
    return mix_Ps


def generate_Ps(n, S, A, gen="random", eps=0.1, seed=250, sparsity=0.05):
    def randomgen(n, nS, nA, seed=250):
        Ps = []
        np.random.seed(seed)
        for _ in range(n):
            sparse = np.random.binomial(1, sparsity, size=(nA * nS, nS))
            for i in range(nA * nS):
                if sum(sparse[i, :]) == 0:
                    sparse[i, np.random.randint(nS)] = 1
            P = sparse * np.random.uniform(0, 1, size=(nA * nS, nS))
            P = P / np.sum(P, 1)[:, np.newaxis]
            Ps.append(P)
        return Ps

    def intergen(n, nS, nA, eps=0.1, seed=250):
        np.random.seed(seed)
        Ps = []
        P = np.random.uniform(0, 1, size=(nA * nS, nS))
        P = P / np.sum(P, 1)[:, np.newaxis]
        for _ in range(n):
            sparse = np.random.binomial(1, 0.05, size=(nA * nS, nS))
            for i in range(nA * nS):
                if sum(sparse[i, :]) == 0:
                    sparse[i, np.random.randint(nS)] = 1
            U = sparse * np.random.uniform(0, 1, size=(nA * nS, nS))
            U = U / np.sum(U, 1)[:, np.newaxis]
            PP = P * (1 - eps) + eps * U
            Ps.append(PP)
        return Ps

    if gen == "random":
        return randomgen(n, S, A, seed)
    else:
        return intergen(n, S, A, eps, seed)
