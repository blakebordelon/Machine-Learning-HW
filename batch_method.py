import numpy as np

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


def batch_newton_raphson(X):
    return
