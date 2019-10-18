import RNN_models as M
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def train_model(model, nepochs, optimizer, taueff, batchsize):

    k = 10
    max_time = 128
    input_dim = 131
    tau = taueff * nepochs
    criterion = nn.NLLLoss()
    print(model.state_dict())
    #optimizer = optim.Adam(model.parameters(), lr = lr)
    model.train()
    print_every = 30
    num_evals = int(nepochs/print_every)+1
    kl_train = 8
    km_train = 9
    kmax = 64

    train_errors = torch.zeros(num_evals)
    test_errors = torch.zeros(num_evals)
    nll_loss = torch.zeros(num_evals)
    for n in range(nepochs):

        with torch.autograd.set_detect_anomaly(False):
            k = np.random.randint(kl_train, km_train)
            #km_train = min(int( (1-1/tau) * km_train + 1/tau * kmax), 128)
            #km_train = min(64, int(km_train + 1/tau*(kmax - km_train) ))
            x = M.generate_batch(k, batchsize, input_dim)
            y = x[k+2:2*k+3, : ,:].argmax(dim=2) # true output
            x.requires_grad = True

            yhat = model.forward_pass(x)

            loss = 0
            for i in range(k+1):
                loss += criterion(yhat[i,:,:], y[i,:])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print("loss at %d = %lf" % (n, loss.item()))

            if n % print_every == 0 and n > 0:
                train_errors[int(n/print_every)] = error_estimate(model, num_samples = 10, batchsize = batchsize, kl = 8, km = 10)
                #test_errors[int(n/print_every)] = error_estimate(model, num_samples = 10, batchsize = batchsize, kl = 65, km = 128)
                nll_loss[int(n/print_every)] = loss.item()
                print("train error = %lf" % train_errors[int(n/print_every)])
                if n > 0:
                    km_train += (64 - 10) / print_every
                    km_train = min(int(km_train), 64)

    return train_errors, test_errors, nll_loss


# error estimate: fine grained
def error_estimate(model, num_samples, batchsize, kl, km):
    error = 0
    for i in range((num_samples)):
        k = np.random.randint(kl,km)
        x = M.generate_batch(k,batchsize, 131)
        y = x[k+2:2*k+3, : ,:].argmax(dim=2)
        yhat = model.forward_pass(x)
        yp = yhat.argmax(dim = 2)[0:k+1,:]
        correct = 1/(k+1) * torch.eq(yp, y).type(torch.float).sum(dim = 0)
        errs = (torch.ones(correct.shape) - correct)
        #error += 1/k * 1/num_samples * errs
        if i == 0:
            for j in range(5):
                print("desired")
                print(M.convert_ints_to_str(y[:,j].type(torch.long), M.words))
                print("received")
                print(M.convert_ints_to_str(yp[:,j].type(torch.long), M.words))
    return error


# get a model based on input parameters
def get_model(model_type, input_dim=131, hidden_dim = 200, stack_dim=200, num_layers=1, batchsize=25):
    if model_type == 'vanilla':
        return M.VanillaRNN(input_dim, hidden_dim, num_layers)
    elif model_type == 'lstm':
        return M.LSTM_seq_cell(input_dim, hidden_dim)
    else:
        return M.LSTM_Stack(input_dim, hidden_dim, stack_dim, batchsize)


def get_optimizer(model, optimizer_type, lr):
    if optimizer_type == 'Adam':
        return optim.Adam(model.parameters(), lr = lr)
    elif optimizer_type == 'RMSprop':
        return optim.RMSprop(model.parameters(), lr = lr)

    elif optimizer_type == 'Adagrad':
        return optim.Adagrad(model.params(), lr =lr)

    return optim.SGD(model.parameters(), lr = lr)



def train_no_tuning(nepochs = 5000, optimizer_type = 'Adam', taueff = 5, lr=1e-3, batchsize = 25):

    model_types = ['vanilla', 'lstm', 'lstm_stack']
    #model_types = ['vanilla']
    all_train = []
    all_test = []
    all_nll = []
    all_models = []
    final_errs = []
    for i in range(len(model_types)):
        m = model_types[i]
        model = get_model(m, batchsize = batchsize)
        optimizer = get_optimizer(model, optimizer_type, lr)
        train_errs, test_errs, nll_loss = train_model(model, nepochs, optimizer, taueff, batchsize = batchsize)
        all_train.append(train_errs)
        all_test.append(test_errs)
        all_nll.append(nll_loss)
        all_models.append(model)
        final_errs.append(test_errs[len(test_errs)-1])
        PATH =  m + '_no_tuning.pt'
        torch.save(model.state_dict(), PATH)

    for i in range(len(model_types)):
        print_every = 30
        xaxis = print_every*np.linspace(0, int(nepochs/print_every), num = int(nepochs/print_every) + 1 )
        for i in range(len(lr_grid)):

            plt.plot(xaxis[0:len(all_train[i])-2], all_train[i][0:len(all_train[i])-1], '--', label = 'train | ' + model_types[i])
            plt.plot(xaxis[0:print_every-1], all_test[i][0:print_every-1], label = 'test | ' + model_types[i])

        plt.xlabel('iterations')
        plt.ylabel('Error')
        plt.legend()
        plt.savefig('error_train_vs_test_notuning.pdf')

        for i in range(len(lr_grid)):

            plt.plot(xaxis[0:len(all_nll[i])], all_nll[i], '--', label = model_types[i])
            #plt.plot(xaxis, test_errs[i], label ='test | lr = %lf' % lr_grid[i])

        plt.xlabel('iterations')
        plt.ylabel('NLLLoss')
        plt.legend()
        plt.savefig('loss_train_vs_test_notuning.pdf')


    return


# experiment with
# learning rate
def search_learning_rate(model_type, nepochs, optimizer_type, taueff, lr_grid, batchsize):
    all_train = []
    all_test = []
    all_nll = []
    all_models = []
    final_errs = []
    for lr in lr_grid:
        model = get_model(model_type, batchsize = batchsize)
        optimizer = get_optimizer(model, optimizer_type, lr)
        train_errs, test_errs, nll_loss = train_model(model, nepochs, optimizer, taueff, batchsize = batchsize)
        all_train.append(train_errs)
        all_test.append(test_errs)
        all_nll.append(nll_loss)
        all_models.append(model)
        final_errs.append(test_errs[len(test_errs)-1])


    best_index = np.argmin(final_errs)
    best_model = all_models[best_index]
    PATH = 'best_model_lr.pt'
    torch.save(best_model.state_dict(), PATH)

    print_every = 30
    xaxis = print_every*np.linspace(0, int(nepochs/print_every), num = int(nepochs/print_every) + 1 )
    for i in range(len(lr_grid)):

        plt.plot(xaxis[0:print_every-1], all_train[i][0:print_every-1], '--', label = 'train | lr = %lf' % lr_grid[i])
        plt.plot(xaxis[0:print_every-1], all_test[i][0:print_every-1], label = 'test | lr = %lf' % lr_grid[i])

    plt.xlabel('iterations')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig('error_train_vs_test.pdf')

    for i in range(len(lr_grid)):

        plt.plot(xaxis[0:len(all_nll[i])], all_nll[i], '--', label ='lr = %lf' % lr_grid[i])
        #plt.plot(xaxis, test_errs[i], label ='test | lr = %lf' % lr_grid[i])

    plt.xlabel('iterations')
    plt.ylabel('NLLLoss')
    plt.legend()
    plt.savefig('loss_train_vs_test.pdf')


    return

def load_and_test_default(PATH, model_type):
    model = get_model(model_type)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    return error_estimate(model, num_samples = 500, batchsize = 20, kl=65, km=128)


def report_errors(errs, mtypes):

    for i in range(len(mtypes)):
        print("model type: " + mtypes[i] + " | test error %lf" % errs[i])
    return

lr_grid = np.linspace(1e-2, 0.1, num = 2)
#search_learning_rate(model_type = 'vanilla', nepochs = 5000, optimizer_type = 'Adam', taueff = 0.25, lr_grid =lr_grid, batchsize = 20)
train_no_tuning()

PATH_end = '_notuning.pt'
mtypes  = ['vanilla', 'lstm', 'lstm_stack']
errs = [load_and_test_default(m + PATH_end, m) for m in mtypes]
report_errors(errs, mtypes)


def search_curricula(model, learning_rate, optimizer, tau_vals):

    train_model(model, )

    return



def search_regularizer():
    return

# experiment with regularizers
# hyper parameters to vary for LSTM
# 1. learning rate / optimizer
# 2. layer number
# 3. hidden-dim
# 4. Learning schedule (curriculum learning etc)  tau dk/dt = k - k_max  (vary tau and get different learning schedules)
# 5. Batch size
# additional params for Stack
# 1. stack_dim
# write method for evaluation of the network ; save model params

def load_file(file_name, model_type, input_dim, hidden_dim, stack_dim, num_layers):

    model = []
    if model_type == 'vanilla':
        model = M.VanillaRNN(input_dim, hidden_dim)

    elif model_type == 'lstm':
        model = M.LSTM_seq_cell(input_dim, hidden_dim)
