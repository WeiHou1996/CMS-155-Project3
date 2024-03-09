import torch
import torch.nn as nn

class LSTM_S(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_S, self).__init__()
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.output_size = output_size #output size
        self.seq_length = 40 #sequence length
        self.num_layers = 1

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, output_size) #fully connected last layer
        self.relu = nn.ReLU()
        self.Softmax = nn.Softmax(dim=1)
    
    def forward(self, input, h0, c0):
        # Propagate input through LSTM with given hidden state
        output, (hn, cn) = self.lstm(input, (h0, c0)) #lstm with input, hidden, and internal state
        output = self.fc_1(output)
        output = self.relu(output)
        output = self.fc(output)

        output_hn = self.fc_1(hn)
        output_hn = self.relu(output_hn)
        output_hn = self.fc(output_hn)
        #output = self.Softmax(output)
        return output_hn, hn, cn
    
    def initHidden(self):
        return torch.zeros(self.num_layers, self.hidden_size)