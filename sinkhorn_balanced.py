# Copyright (c) 2020,2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np

def sinkhorn_stabilized(a, b, M, reg, numItermax=100, tau=1e3, stopThr=1e-9, print_period=20):
    """
    Solve the entropic regularization OT problem with log stabilization
    The function solves the following optimization problem:
    .. math::
        \gamma = arg\min_\gamma <\gamma,M>_F + reg\cdot\Omega(\gamma)
        s.t. \gamma 1 = a
             \gamma^T 1= b
             \gamma\geq 0
    where :
    - M is the (ns,nt) metric cost matrix
    - :math:`\Omega` is the entropic regularization term :math:`\Omega(\gamma)=\sum_{i,j} \gamma_{i,j}\log(\gamma_{i,j})`
    - a and b are source and target weights (sum to 1)
    The algorithm used for solving the problem is the Sinkhorn-Knopp matrix
    scaling algorithm as proposed in [2]_ but with the log stabilization
    proposed in [10]_ an defined in [9]_ (Algo 3.1) .
    Parameters
    ----------
    a : np.ndarray (ns,)
        samples weights in the source domain
    b : np.ndarray (nt,)
        samples in the target domain
    M : np.ndarray (ns,nt)
        loss matrix
    reg : float
        Regularization term >0
    tau : float
        thershold for max value in u or v for log scaling
    stopThr : float, optional
     Stop threshol on error (>0)
    Returns
    -------
    gamma : (ns x nt) ndarray
        Optimal transportation matrix for the given parameters
    log : dict
        log dictionary return only if log==True in parameters
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    # init data
    na = len(a)
    nb = len(b)

    # we assume that no distances are null except those of the diagonal of
    # distances

    alpha, beta = np.zeros(na), np.zeros(nb)

    u, v = np.ones(na) / na, np.ones(nb) / nb

    def get_K(alpha, beta):
        """log space computation"""
        return np.exp(-(M - alpha.reshape((na, 1))
                        - beta.reshape((1, nb))) / reg)

    def get_Gamma(alpha, beta, u, v):
        """log space gamma computation"""
        return np.exp(-(M - alpha.reshape((na, 1)) - beta.reshape((1, nb)))
                      / reg + np.log(u.reshape((na, 1))) + np.log(v.reshape((1, nb))))

    K = get_K(alpha, beta)
    transp = K
    loop = 1
    cpt = 0
    err = 1
    while loop:
        uprev = u
        vprev = v

        # sinkhorn update
        v = b / (np.dot(K.T, u) + 1e-16)
        u = a / (np.dot(K, v) + 1e-16)

        # remove numerical problems and store them in K
        if np.abs(u).max() > tau or np.abs(v).max() > tau:
            alpha, beta = alpha + reg * np.log(u), beta + reg * np.log(v)
            u, v = np.ones(na) / na, np.ones(nb) / nb
            K = get_K(alpha, beta)

        if cpt % print_period == 0:
            # we can speed up the process by checking for the error only all
            # the 10th iterations

            transp = get_Gamma(alpha, beta, u, v)
            err = np.linalg.norm((np.sum(transp, axis=0) - b))**2

        if err <= stopThr:
            loop = False

        if cpt >= numItermax:
            loop = False

        if np.any(np.isnan(u)) or np.any(np.isnan(v)):
            # we have reached the machine precision
            # come back to previous solution and quit loop
            print('Warning: numerical errors at iteration', cpt)
            u = uprev
            v = vprev
            break
        # print(cpt)
        cpt = cpt + 1

    return get_Gamma(alpha, beta, u, v)
