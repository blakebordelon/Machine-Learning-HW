import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import string



def tensor_to_str(x):
    ind = x.argmax()
    return int_to_str([ind])

def int_to_str(indices):
    word = ''
    for i in indices:
        word += string.ascii_lowercase[i]
    return word

num_words = 128
alphabet_size = 26
num_char = 6
words = [int_to_str(np.random.randint(26, size = 6)) for i in range(num_words)]
print(words)
words.insert(0, '!')
words.insert(0,'&')
words.insert(0,'*')

string_dict = {words[i] : i for i in range(len(words))}

def convert_ints_to_str(indices, words):
    sentence = ''
    for i in indices:
        sentence += ' ' + words[i]
    return sentence

# tensor should be k x input_dim
def convert_tensors_to_str(x, words):
    return convert_ints_to_str(x.argmax(dim=1), words)


input_dim = len(words)

# one hot encoding
def generate_batch(k, batchsize, input_dim):
    batch = torch.zeros(2*k+3, batchsize, input_dim)
    # special symbols at beginning middle and end
    batch[0,:,0] = torch.ones(batchsize)
    batch[k+1,:,1] = torch.ones(batchsize)
    batch[2*k+2,:,2] = torch.ones(batchsize)
    symb_indices = np.random.randint(3,input_dim, size =(k,batchsize))
    for b in range(batchsize):
        for i in range(k):
            s = symb_indices[i,b]
            batch[i+1,b,s] = 1
            batch[2*k+1-i,b,s] = 1

    return batch


class VanillaRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encode_RNN = nn.RNN(input_dim, hidden_dim)
        self.decode_RNN = nn.RNN(input_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, input_dim)
        self.sm = nn.LogSoftmax(dim = 1)
        return


    def forward_pass(self, x):
        k = int( (x.shape[0] - 3)/2)
        outputs = torch.zeros(k ,x.shape[1], x.shape[2])
        out1, h1 = self.encode_RNN(x[0:k+2,:,:])
        out2, h2 = self.decode_RNN(torch.zeros(k, x.shape[1], x.shape[2]), h1)
        return self.sm(self.readout(out2))



class LSTM_seq_cell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encode_LSTM = nn.LSTMCell(input_dim, hidden_dim, num_layers = 2)
        self.decode_LSTM = nn.LSTMCell(input_dim, hidden_dim, num_layers = 2)
        self.readout = nn.Linear(hidden_dim, input_dim)
        self.sm = nn.LogSoftmax(dim = 1)
        return

    def forward_pass(self, x):
        k = int( (x.shape[0] - 3)/2)
        outputs = torch.zeros(k ,x.shape[1], x.shape[2])
        h = torch.zeros(x.shape[1], self.hidden_dim)
        c = torch.zeros(x.shape[1], self.hidden_dim)
        x0 = torch.zeros(x.shape[1], x.shape[2])
        for i in range(k+2):
            (h,c) = self.encode_LSTM(x[i,:,:], (h,c))
        for i in range(k):
            (h,c) = self.decode_LSTM(x0, (h,c))
            outputs[i, : , :] = self.sm(self.readout(h))
        return outputs



class LSTM_Stack(nn.Module):
    def __init__(self, input_dim, hidden_dim, stack_dim, batch_size, max_t):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.max_t = max_t
        self.stack_dim = stack_dim
        self.encode_LSTM = nn.LSTMCell(input_dim + stack_dim, hidden_dim)
        self.decode_LSTM = nn.LSTMCell(input_dim + stack_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, input_dim + stack_dim + 2)
        self.sm = nn.LogSoftmax(dim = 1)
        self.V = torch.zeros(max_t, batch_size, stack_dim)
        self.s = torch.zeros(max_t, batch_size)
        self.d = torch.zeros(batch_size)
        self.u = torch.zeros(batch_size)

        return

    def forward_pass(self, x):
        k = int( (x.shape[0] - 3)/2)
        self.zero_out_stack()
        outputs = torch.zeros(k ,x.shape[1], x.shape[2])
        h = torch.zeros(x.shape[1], self.hidden_dim)
        c = torch.zeros(x.shape[1], self.hidden_dim)
        x0 = torch.zeros(x.shape[1], x.shape[2])
        rt = torch.zeros(self.batch_size, self.stack_dim)
        for i in range(k+2):
            z = torch.cat((x[i,:,:], rt), dim = 1)
            (h,c) = self.encode_LSTM(z, (h,c))
            out = self.readout(h)
            ot, vt, dt, ut = out.split([input_dim, stack_dim, 1, 1], dim = 1)
            rt = self.update_stack(vt, dt, ut, i)
        for i in range(k):
            z = torch.cat((x0, rt), dim = 1)
            (h,c) = self.decode_LSTM(z, (h,c))
            out = self.readout(h)
            ot, vt, dt, ut = out.split([input_dim, stack_dim, 1, 1], dim = 1)
            rt = self.update_stack(vt, dt, ut, i+k+2)
            outputs[i,:,:] = self.sm(ot)
        return outputs

    def zero_out_stack(self):
        self.V = torch.zeros(self.max_t, self.batch_size, self.stack_dim)
        self.s = torch.zeros(self.max_t, self.batch_size)
        self.d = torch.zeros(self.batch_size)
        self.u = torch.zeros(self.batch_size)
        return

    def update_stack(self, vt, dt, ut, t):

        self.V[t,:,:] = vt * 1.0
        self.s[t,:] = dt[:,0]* 1.0

        rt = torch.zeros(self.batch_size, self.stack_dim)
        del_score = ut[:,0]* 1.0
        left_compare = torch.zeros(self.batch_size)
        #s = self.s.clone()
        for i in range(t-1, -1, -1):
            self.s[i,:] = torch.max(left_compare,  torch.max(left_compare, 1.0 * del_score))
            #s[i,:] = torch.max(left_compare,  torch.max(left_compare, 1.0 * del_score))
            del_score += -self.s[i,:]*1.0

        pop_score = torch.ones(self.batch_size)
        c = torch.zeros(1,self.batch_size)
        V = self.V.clone()
        for i in range(t, -1, -1):
            coeff = torch.min( self.s[t,:].clone() , torch.max( left_compare,  pop_score))
            #coeff = torch.min( s[t,:] , torch.max( left_compare,  pop_score))
            """
            for j in range(self.stack_dim):
                #rt[:,j] +=  coeff * self.V[t,:,j] * 1.0
                rt[:,j] += coeff * V[t,:,j]
            """
            A = coeff.repeat(self.stack_dim, 1)
            rt += A.transpose(0,1) * V[t,:,:] * 1.0
            #rt += c * q
        #self.s = s.clone()
        return rt


# hyper parameters to vary for LSTM
# 1. learning rate 2. layer number 3. hidden dim 4. Learning schedule (curriculum etc)
# additional params for Stack 1. stack_dim
# write method for evaluation of the network ; save model params

batchsize = 75
k = 10
hidden_dim = 200
stack_dim = 40
num_layers = 1
max_time = 128
#model = LSTM_seq_cell(input_dim, hidden_dim)
#model = LSTM_Stack(input_dim, hidden_dim, stack_dim, batchsize, max_time)
model = VanillaRNN(input_dim, hidden_dim)
criterion = nn.NLLLoss()
lr = 8e-3
print(model.state_dict())
optimizer = optim.Adam(model.parameters(), lr = lr)
nepochs = 1000

kl_train = 8
km_train = 9


for n in range(nepochs):

    with torch.autograd.set_detect_anomaly(False):
        k = np.random.randint(kl_train, km_train)
        x = generate_batch(k, batchsize, input_dim)
        y = x[k+2:2*k+2, : ,:].argmax(dim=2)
        x.requires_grad = True
        yhat = model.forward_pass(x)

        loss = 0
        for i in range(k):
            loss += criterion(yhat[i,:,:], y[i,:])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(loss.item())


print(convert_tensors_to_str(yhat[:,0,:], words))
print(convert_ints_to_str(y[:,0], words))
