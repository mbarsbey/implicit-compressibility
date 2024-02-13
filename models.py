# Two layer fully connected neural network

import torch
import torch.nn as nn
import copy

class MultiLayerNN(nn.Module):

    def __init__(self, input_dim=28*28, width=50, depth=2, num_classes=10, activation="relu"):
        assert depth >= 2
        super(MultiLayerNN, self).__init__()
        self.input_dim = input_dim
        self.width = width
        self.num_classes = num_classes
        
        layers = []
        layers.append(nn.Linear(self.input_dim, self.width, bias=False))
        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.Sigmoid())
        for i in range(depth-2):
            layers.append(nn.Linear(self.width, self.width, bias=False))
            if activation == "relu":
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.Sigmoid())
        layers.append(nn.Linear(self.width, self.num_classes, bias=False))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), self.input_dim)
        x = self.fc(x)
#        x = x / self.width # this has been disabled to make lr meaningful in an absolute fashion
        return x
    

if __name__ == '__main__':
    x = torch.randn(5, 1, 32, 32)
    net = MultiLayerNN(input_dim=32*32, width=123)
    print(net(x))

class Net(nn.Module):
    def __init__(self, d_input, d_output, nb_neurons=500):
        super(Net, self).__init__()
        self.n = nb_neurons
        self.input = nn.Linear(d_input, self.n, bias=False)  # input layer
        self.output = nn.Linear(self.n, d_output, bias=False)    # output layer
        # self.output.weight.data.fill_(0)
    def forward(self, x, device=torch.device('cpu')):
        x = self.input(x)
        x = torch.relu(x)
        x = self.output(x)
        return x