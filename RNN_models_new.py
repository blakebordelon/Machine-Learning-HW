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
    def __init__(self, input_dim, hidden_dim, stack_dim, batch_size, max_t = 2*128+3):
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
        self.relu = nn.ReLU()
        self.matrix_partial_sums = torch.zeros(self.batch_size, self.max_t, self.max_t)
        return

    def init_matrix_partial_sum(self):
        matrix_partial_sums = torch.ones(self.batch_size, self.max_t, self.max_t)
        for i in range(self.max_t):
            matrix_partial_sums[:,i,0:i+1] = torch.zeros(i+1)
        self.matrix_partial_sums =  matrix_partial_sums
        return
    def forward_pass(self, x):
        l = x.shape[0]
        k = int( (l-3) /2 )
        self.init_matrix_partial_sum()
        # Value matrix and strength vector
        V = torch.zeros(self.batch_size, self.max_t, self.stack_dim)
        s = torch.zeros(self.batch_size, self.max_t)
        # hidden vector and control vector for LSTM
        h = torch.zeros(self.batch_size, self.hidden_dim)
        c = torch.zeros(self.batch_size, self.hidden_dim)

        # output of stack
        rt = torch.zeros(self.batch_size, self.stack_dim)
        outputs = torch.zeros(k + 2, self.batch_size, self.input_dim)
        for i in range(k + 1):
            (h,c) =self.encode_LSTM( torch.cat((x[i,:,:], rt), dim = 1) , (h,c))
            out = self.readout(h)
            vt, ot, ut, dt = out.split([self.stack_dim, self.input_dim, 1,1], dim = 1)
            V, s, rt = self.update_stack(V, s, vt, self.sigm(ut), self.sigm(dt), i)

        for i in range(k+2):
            (h,c) =self.encode_LSTM( torch.cat((torch.zeros(self.batch_size, self.input_dim), rt), dim = 1) , (h,c))
            out = self.readout(h)
            vt, ot, ut, dt = out.split([self.stack_dim, self.input_dim, 1,1], dim = 1)
            V, s, rt = self.update_stack(V, s, vt, self.sigm(ut), self.sigm(dt), i + k + 1)
            outputs[i,:,:] = self.sm(ot)
        return outputs

    def update_stack(self, V, s, vt, ut, dt, t):
        # batch x k x stack dim

        zeromat = torch.zeros(V.shape)
        zeromat[:,t,:] = vt
        Vp = V + zeromat


        prod = torch.baddbmm(torch.zeros(s.unsqueeze(-1).shape), self.matrix_partial_sums[:,0:s.shape[1],0:s.shape[1]], s.unsqueeze(-1))
        zero_vals = torch.zeros(s.shape)

        prod = prod.squeeze()
        new_strength = ut.repeat(1, s.shape[1]) -  prod

        sp = torch.max(zero_vals, s - torch.max(zero_vals, new_strength))
        #sp = torch.cat((sp, dt), dim = 1)
        sp[:,t] = dt[:,0]
        rt = torch.zeros(self.batch_size, self.stack_dim)
        inner_max_vec = self.relu(torch.ones(prod.shape) - prod - sp)
        matrix_A = torch.min(sp, inner_max_vec).unsqueeze(1)

        rt = torch.baddbmm(torch.zeros(self.batch_size, 1, self.stack_dim), matrix_A, Vp).squeeze()

        return Vp, sp, rt
