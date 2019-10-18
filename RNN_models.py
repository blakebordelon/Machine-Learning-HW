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

#def convert_ints_to_str(indices, words):
def convert_ints_to_str(indices, words):
    sentence = ''
    for i in indices:
        sentence += ' ' + words[int(i)]
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


# try to have the nn decide its own output length
class VanillaRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encode_RNN = nn.RNNCell(input_dim, hidden_dim)
        self.decode_RNN = nn.RNNCell(input_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, input_dim)
        self.sm = nn.LogSoftmax(dim = 1)
        return


    def forward_pass(self, x):
        k = int( (x.shape[0] - 3)/2)
        outputs = torch.zeros(k+1 ,x.shape[1], x.shape[2])
        batch_done = torch.zeros(x.shape[1]) # checks whether a given element has encountered the end of sentence symbol (index 2)
        h = torch.zeros(x.shape[1], self.hidden_dim)
        begin_char = torch.zeros(x.shape[1], self.input_dim)
        begin_char[:,0] = torch.ones(x.shape[1])
        outputs[:,:,0] = torch.ones(outputs.shape[0], x.shape[1])

        for i in range(k+2):
            h = self.encode_RNN(x[i,:,:], h)
        for i in range(k+1):
            h = self.decode_RNN(torch.zeros(x.shape[1], x.shape[2]), h)
            out = self.sm(self.readout(h))
            #outputs[i,:,:] = out
            outp = out.argmax(dim =1)
            z = (outp == (2*torch.ones(x.shape[1]).type(torch.long)) ).type(torch.float) # check if end of sentence
            batch_done = torch.max(batch_done, z)
            b = batch_done.repeat(self.input_dim,1).transpose(0,1)
            outputs[i,:,:] = (torch.ones(b.shape) - b) * out + b*begin_char
            if batch_done.sum() == x.shape[1] or i > 2*k+5:
                return outputs
        return outputs



class LSTM_seq_cell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.encode_LSTM = nn.LSTMCell(input_dim, hidden_dim)
        self.decode_LSTM = nn.LSTMCell(input_dim, hidden_dim)
        self.readout = nn.Linear(hidden_dim, input_dim)
        self.sm = nn.LogSoftmax(dim = 1)
        return


    def forward_pass(self, x):
        k = int( (x.shape[0] - 3)/2)
        outputs = torch.zeros(128, x.shape[1], x.shape[2])
        h = torch.zeros(x.shape[1], self.hidden_dim)
        c = torch.zeros(x.shape[1], self.hidden_dim)
        x0 = torch.zeros(x.shape[1], x.shape[2])
        batch_done = torch.zeros(x.shape[1]) # checks whether a given element has encountered the end of sentence symbol (index 2)
        begin_char = torch.zeros(x.shape[1], self.input_dim)
        begin_char[:,0] = torch.ones(x.shape[1])
        for i in range(k+2):
            (h,c) = self.encode_LSTM(x[i,:,:], (h,c))
        for i in range(k + 1):
            (h,c) = self.decode_LSTM(x0, (h,c))
            out = self.sm(self.readout(h))
            outp = out.argmax(dim =1)
            z = (outp == (2*torch.ones(x.shape[1]).type(torch.long)) ).type(torch.float) # check if end of sentence
            batch_done = torch.max(batch_done, z)
            b = batch_done.repeat(self.input_dim,1).transpose(0,1)
            outputs[i,:,:] = (torch.ones(b.shape) - b) * out + b*begin_char
            if batch_done.sum() == x.shape[1]:
                return outputs
        return outputs



class LSTM_Stack(nn.Module):
    def __init__(self, input_dim, hidden_dim, stack_dim, batch_size, max_t = 129):
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
        self.sigm = nn.Sigmoid()
        self.V = torch.zeros(max_t, batch_size, stack_dim)
        self.s = torch.zeros(max_t, batch_size)

        return

    def forward_pass(self, x):
        k = int( (x.shape[0] - 3)/2)
        self.zero_out_stack()
        outputs = torch.zeros(k+1, x.shape[1], x.shape[2])
        outputs[:,:,0] = 1.0*torch.ones(k+1, x.shape[1]) # initialize null response to "begin of sentence" symbol
        h = torch.zeros(x.shape[1], self.hidden_dim)
        c = torch.zeros(x.shape[1], self.hidden_dim)
        x0 = torch.zeros(x.shape[1], x.shape[2])
        rt = torch.zeros(x.shape[1], self.stack_dim)
        batch_done = torch.zeros(x.shape[1]) # checks whether a given element has encountered the end of sentence symbol (index 2)
        begin_char = torch.zeros(x.shape[1], self.input_dim)
        begin_char[:,0] = torch.ones(x.shape[1])
        for i in range(k+2):

            z = torch.cat((x[i,:,:], rt), dim = 1)
            (h,c) = self.encode_LSTM(z, (h,c))
            out = self.readout(h)
            ot, vt, dt, ut = out.split([self.input_dim, self.stack_dim, 1, 1], dim = 1)
            rt = self.update_stack(vt, self.sigm(dt), self.sigm(ut), i)
        for i in range(k+1):
            z = torch.cat((x0, rt), dim = 1)
            (h,c) = self.decode_LSTM(z, (h,c))
            all_out = self.readout(h)
            ot, vt, dt, ut = all_out.split([self.input_dim, self.stack_dim, 1, 1], dim = 1)
            rt = self.update_stack(vt, self.sigm(dt), self.sigm(ut), i+k+2)
            out = self.sm(ot)
            outputs[i,:,:] = out

            outp = out.argmax(dim=1)


            b = batch_done.repeat(self.input_dim,1).transpose(0,1)
            outputs[i,:,:] = (torch.ones(b.shape) - b) * out + b*begin_char

            z = (outp == (2*torch.ones(x.shape[1]).type(torch.long)) ).type(torch.float) # check if end of sentence

            batch_done = torch.max(batch_done, z)


        return outputs

    def zero_out_stack(self):
        self.V = torch.zeros(self.max_t, self.batch_size, self.stack_dim)
        self.s = torch.zeros(self.max_t, self.batch_size)
        return

    def get_partial_sums(self,t):
        z = torch.zeros(self.s.shape[1], self.s.shape[0],1)

        matrix_partial_sums = torch.ones(self.batch_size, self.max_t, self.max_t)
        for i in range(t):
            matrix_partial_sums[:,i,0:i+1] = torch.zeros(matrix_partial_sums[:,i,0:i+1].shape)


        z[:,:,0] = self.s.transpose(0,1)
        input = torch.zeros(z.shape)

        # calculate partial sums of strength vector
        partials = torch.baddbmm(input, matrix_partial_sums, z)

        partials.reshape((partials.shape[0], partials.shape[1]))
        return partials

    def update_stack(self, vt, dt, ut, t):

        # append new vector to the value matrix
        self.V[t,:,:] = vt * 1.0


        left_compare = torch.zeros(self.batch_size)
        #s = self.s.clone()


        if t > 0:

            partials = self.get_partial_sums(t)
            ut_sub = ut.repeat(1, self.max_t)
            z2 = torch.zeros(partials.shape)
            z2[:,:,0] = ut_sub
            # compute the new strength vector
            copy_s_ext = torch.zeros(partials.shape)

            x = torch.max(torch.zeros(partials.shape), self.s.reshape(partials.shape) - torch.max(torch.zeros(partials.shape), z2 - partials) )
            self.s = x[:,:,0].transpose(0,1)

        # append the strength dt to the t-th element of the strength vector
        self.s[t,:]= dt[:,0] * 1.0




        pop_score = torch.ones(self.batch_size)
        c = torch.zeros(1,self.batch_size)
        A = torch.zeros(self.V.shape[0], self.batch_size, self.stack_dim)

        partials2 = self.get_partial_sums(t)
        a = torch.max(torch.zeros(partials2.shape), torch.ones(partials2.shape) - partials2)

        s = self.s.clone()
        coeff =torch.min(s.transpose(0,1), a[:,:,0])
        c_ext = torch.zeros(coeff.shape[0], coeff.shape[1],1)
        c_ext[:,:,0] = coeff
        c_ext = c_ext.transpose(1,2)


        Q = self.V.clone()
        Q = Q.transpose(0,1)

        input = torch.zeros(c_ext.shape[0], c_ext.shape[1], Q.shape[2])

        rt = torch.baddbmm(input, c_ext, Q).transpose(1,2)
        rt = rt.reshape(rt.shape[0], rt.shape[1])


        #rt = coeff * self.V

        return rt


# to dos
# 1. Create an interface for these models in another file
# 2. Learn how to save models
# 3. Create a Language class that stores our dictionary and can convert tensors to strings
# 3.


# hyper parameters to vary for LSTM
# 1. learning rate / optimizer
# 2. layer number
# 3. hidden-dim
# 4. Learning schedule (curriculum learning etc)
# 5. Batch size
# additional params for Stack
# 1. stack_dim
# write method for evaluation of the network ; save model params

"""
batchsize = 75
k = 10
hidden_dim = 200
stack_dim = 40
num_layers = 2
max_time = 128
#model = LSTM_seq_cell(input_dim, hidden_dim)
#model = LSTM_Stack(input_dim, hidden_dim, stack_dim, batchsize, max_time)
model = VanillaRNN(input_dim, hidden_dim, num_layers)
criterion = nn.NLLLoss()
lr = 1e-3
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
"""
