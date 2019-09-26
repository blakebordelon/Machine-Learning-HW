import numpy as np
import timeit
# calculate F1 on test set as a function of training time


def grad_total(X, y, w):
    P,N = X.shape
    g = np.zeros(N)
    for n in range(P):
        g += - y[n] * x[n,:] / (1+np.exp(-y[n] * np.dot(x[n,:],w)))
    return g

def hessian_total(X,y,w):
    P,N = X.shape
    H = np.zeros((N,N))
    for n in range(P):
        x = X[n,:]
        H +=  np.outer(x,x) * np.exp(-y[n] * np.dot(x,w)) / (1+np.exp(-y[n]*np.dot(w,x)))**2
    return H


def batch_newton_raphson(X, y, Xte, yte, num_iter):

    P, N = X.shape
    P_te, N_te = Xte.shape
    w = np.zeros(N)
    for t in range(num_iter):

        g = grad_total(X, y, w)
        H = hessian_total(X,y,w)
        w += - np.linalg.inv(H) @ g
        tp = 0
        fn = 0
        fp = 0
        for n in range(P_te):
            x = Xte[n,:]
            y = yte[n]
            p = predict(w,x,y)
            



    return
