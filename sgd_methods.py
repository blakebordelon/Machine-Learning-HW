import numpy as np
import csv
import matplotlib.pyplot as plt

# hello
# format the data; feature transform and approximate whitening
def format_data(data):
    P, N = data.shape
    X0 = preprocess(data[:,1:N])
    #X0 = data[:,1:N]
    y = data[:,0]
    y = 2*y - np.ones(len(y))
    print(y)
    print(X0.shape)
    X = np.concatenate((X0, np.reshape(np.ones(P), (P,1))), axis = 1)
    print(X.shape)
    return [X, y]

def preprocess(X0):
    X0 = approximate_whitening(X0)

    return X0

# make sure the variance of each feature is 1 and the mean is 0.
def approximate_whitening(X0):
    P,N = X0.shape
    mean = np.mean(X0, axis = 0)
    std = np.std(X0, axis = 0)
    X_zeromean = X0 - np.outer(np.ones(P), mean)
    print(np.mean(X_zeromean, axis = 0))
    X = np.multiply(X_zeromean, np.power(np.outer(np.ones(P),std), -1))
    print("mean and std")
    print(np.mean(X, axis = 0))
    print(np.std(X,axis = 0))
    return X

def read_data(isTrainingSet):
    data = []
    str = 'snow_shoveling_'
    if isTrainingSet==True:
        str += 'train.csv'
    else:
        str += 'test.csv'

    with open(str) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)

    return np.array(data).astype('float')

# calculate F1 statistic
def F1(tp, fp, fn):

    precision = 0
    recall = 0
    if tp + fp ==0:
        precision = 1e10
    else:
        precision = tp / (tp + fp)
    if tp + fn == 0:
        recall = 1e10

    else:
        recall = tp / (tp + fn)

    if precision*recall == 0:
        return 0
    else:
        return 2*precision*recall / (precision + recall)

def loss(w, x, y):
    return np.log(1+np.exp(-y * np.dot(w,x)))

def predict(w,x):
    p_plus = 1 / (1+np.exp(- np.dot(w,x)))
    if p_plus > 0.5:
        return 1
    else:
        return -1

def grad(w,x,y):
    return - y * x / (1+np.exp(-y*np.dot(w,x)))

# options: 1. fixed learning rate 2. decaying learning rate 3. polyak ruppert averaging 4. adagrad
def sgd(X, y, Xte, yte, eta, num_iter, m, option):
    P, N = X.shape
    Pte, Nte = Xte.shape
    tp = 0
    fn = 0
    fp = 0
    w_avg = np.zeros(N)
    w = np.zeros(N)
    num_evals = int(num_iter/m)
    F1_scores = np.zeros(num_evals)
    G = np.zeros((N,N))
    for t in range(num_iter):
        n = np.random.randint(1,P)
        x_n = X[n,:]
        y_n = y[n]


        if t % m == 0 and t!=0:
            tp = 0
            fp = 0
            fn = 0
            for mu in range(Pte):
                x_mu = Xte[mu,:]
                y_mu = yte[mu]
                p=0
                if option != 'polyak_ruppert':
                    p = predict(w,x_mu)
                else:
                    p = predict(w_avg, x_mu)

                if y_mu ==1 and p==1:
                    tp += 1
                elif y_mu == 1 and p!=1:
                    fn += 1
                elif y_mu != 1 and p==1:
                    fp+=1
            F1_scores[int(t/m)] = F1(tp,fp,fn)


        g = grad(w,x_n,y_n)
        G += np.outer(g,g)
        if option == 'fixed':
            w += - eta * g
        elif option == 'decay':
            w += - eta/(t+1) * g
        elif option == 'polyak_ruppert':
            w += -eta * g
            if t==0:
                w_avg = w
            else:
                w_avg = (t) / (t+1) * w_avg + 1/(t+1) * w
        elif option == 'adagrad':
            w += - eta * np.dot( np.power(np.diag(G),-0.5), g)

        else:
            print('option unsupported')
            return

    return F1_scores



Xtr, ytr = format_data(read_data(True))
Xte, yte = format_data(read_data(False))
eta = 1e-4
num_iter = 1000
m = 50

F1_scores_fixed = sgd(Xtr, ytr, Xte, yte, eta, num_iter, m, 'fixed')
F1_scores_decay = sgd(Xtr, ytr,Xte, yte, eta, num_iter, m, 'decay')
F1_scores_polyak = sgd(Xtr, ytr,Xte, yte, eta, num_iter, m, 'polyak_ruppert')
F1_scores_adagrad = sgd(Xtr, ytr,Xte, yte, eta, num_iter, m, 'adagrad')

iters = np.linspace(1,num_iter, num = len(F1_scores_fixed))
plt.semilogx(iters, F1_scores_fixed, label = 'fixed')
plt.semilogx(iters, F1_scores_decay, label = 'decay')
plt.semilogx(iters, F1_scores_polyak, label = 'polyak ruppert')
plt.semilogx(iters, F1_scores_adagrad, label = 'adagrad')

plt.xlabel('iterations')
plt.ylabel('F1')
str = 'F1_all.pdf'
plt.legend()

plt.savefig(str)
plt.show()
