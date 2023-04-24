# Copyright (c) 2020,2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import autograd.numpy as np
import matplotlib.pyplot as plt
import sys
import pdb
import sinkhorn_balanced as sb
sys.path.append('./pymanopt/')
from pymanopt import Problem
from pymanopt.solvers import TrustRegions, ConjugateGradient, SteepestDescent
from pymanopt.manifolds import Stiefel, PositiveDefinite, Product, Euclidean, Grassmann, Sphere
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

np.random.seed(seed=1234)

def temporal_hinge(UtX, S, eta): # UtX is pxn
    F = -(np.diff((UtX * UtX).sum(0)) - eta + S*S) # <UtX, MUtX>_t - <UtX, MUt_X>_{t+1} +eta - \xi_ij
    hinge = np.maximum(F,0)**2.0
    cost = hinge.sum()
    return cost

def compute_U_objective(U, S, X, Y, XX, Pi, lambda_val, eta, slack_penalty=100, pca_val=.1):
    sym = lambda z: z+z.transpose()
    Ut = U.transpose()
    UUtX = np.matmul(U, np.matmul(Ut, X))
    UUtXX = np.matmul(U, np.matmul(Ut, XX))
    UtX = np.matmul(Ut, X)
    n = float(X.shape[1])
    nn = float(XX.shape[1])

    YPit = np.matmul(Y, Pi)
    cost = 0
    cost1 = -(np.linalg.norm(UUtXX - YPit, 'fro')**2.0)/(2*nn) # contrastive objectivee
    cost2 =  (np.linalg.norm(UUtX - X, 'fro')**2.0)/(2*n)
    cost3 = temporal_hinge(UtX, S, eta)/(2*n) # temporal cost.
    cost4 = (S*S).sum()/(2*n) # cost non-negative slacks.
    cost = cost1 + pca_val*cost2 + lambda_val*cost3 + slack_penalty*cost4
    return cost

def meta_solver(X, Y=None, nY=100, p=2, eta=1., lambda_val=1, pca_val=1.0, max_iter=10, U_init=None, use_OT=True, num_iter=5):
    d = X.shape[0]
    nX = X.shape[1]
    if nX < p:
        p = nX

    if Y is None:
        sigma = X.max()
        Y = normalize(np.maximum(0.,X-np.random.rand(X.shape[0],X.shape[1])*sigma),axis=0)
    nY = Y.shape[1]
    if nY > nX:
        XX = np.tile(X, (nY//nX)) # repeat X k times to match nY.
    else:
        XX = X

    manifold_U = Grassmann(d, p)
    manifold_Slack = Euclidean(nX-1) # slack
    manifold = Product((manifold_U, manifold_Slack))

    marginal_X = np.ones(nY)/nY # This is nY because nY = nX.
    marginal_Y = np.ones(nY)/nY

    # Pi is the transport matrix. For now, lets assume its identity.
    def compute_Pi(Z):
        OT = True
        if OT == True:
            if Z is None:
                Z = XX
            dist_matrix = cdist(Y.transpose(), Z.transpose())
            dist_matrix = dist_matrix/dist_matrix.max()
            Pi = sb.sinkhorn_stabilized(marginal_Y, marginal_X, dist_matrix, 0.0001, numItermax=1000, tau=1e3, stopThr=1e-9, print_period=1)
            Pi = generate_coupling_matrix(Pi)
            return Pi
        else:
            return np.eye(nY)[:,:nX]

    def U_Init(p):
        U,_,_ = np.linalg.svd(X, full_matrices=False) #X[:,:p] #np.linalg.qr(X)
        S = np.random.rand(nX-1,)
        return (U[:,:p], S)

    def compute_full_objective(U, S, Pi):
        return compute_U_objective(U, S, X, XX, Y, Pi, lambda_val, eta, pca_val=pca_val)

    # create a permutation matrix from the sinkhorn matrix.
    def generate_coupling_matrix(P):
        Z = np.zeros(P.shape)
        Q = P.argmax(1) # max along the columns.
        for t in range(P.shape[0]):
            Z[t,Q[t]] = 1.
        return Z
    if use_OT:
        Pi = compute_Pi(None)
    else:
        Pi = np.eye(nY)#[:,:nX]

    if U_init is None:
        U_init = U_Init(p)

    obj_prev = 100000.
    tt_mpt, tt_ot = 0., 0.
    for ii in range(max_iter):
        #p = max_iter - ii
        tt = time.time()
        U, S = solve_for_U(X, Y, XX, Pi, p, lambda_val, pca_val, eta, manifold, U_init, num_iter)
        tt_mpt += time.time()-tt

        tt = time.time()
        if max_iter>1:
            UUtXX = np.matmul(U, np.matmul(U.transpose(), XX))
            Pi = compute_Pi(UUtXX)
            obj = compute_full_objective(U, S, Pi)
            print('meta solver: iter:%d obj=%f, sum(Pi)=%f' % (ii, obj, Pi.sum()))
        tt_ot += time.time() - tt

        if max_iter>1:
            if np.abs(obj-obj_prev) < 1e-5:
                break
            obj_prev = obj

    if p > 1:
        U = np.matmul(U, np.matmul(U.transpose(), X)) #.mean(1)[:, np.newaxis]
        U = U/np.linalg.norm(U, axis=0)[np.newaxis, :]
        U = U.mean(1)[:, np.newaxis]
    return U.reshape(-1,1), S, Pi


def solve_for_U(X, Y, XX, Pi, p, lambda_val, pca_val, eta, manifold, U_init, num_iter=20):
    def cost(Z):
        U, S = Z
        return compute_U_objective(U, S, X, Y, XX, Pi, lambda_val, eta, pca_val=pca_val)

    problem = Problem(manifold=manifold, cost=cost, verbosity=0)
    try:
        solver = ConjugateGradient(maxiter=num_iter, logverbosity=0)
        ret = solver.solve(problem, U_init)#[0] #, x=init)
    except:
        print('conjugate gradient crashed! trying TrustRegions...')
        solver = TrustRegions(maxiter=5, logverbosity=0)
        ret = solver.solve(problem, U_init)

    U, S = ret
    return U, S
