import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class reward_Net(nn.Module):
    def __init__(self, x_size, a_size):
        super(reward_Net, self).__init__()
        self.x_size = x_size
        self.a_size = a_size
        self.temp = 1
        # self.linear = nn.Linear(x_size, a_size)
        self.input_fc = nn.Linear(x_size, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.linear = nn.Linear(100, a_size)


    # def forward(self, x):
    #     x = Variable(x.view(-1, self.x_size))
    #     x = self.linear(x)
    #     output = torch.sigmoid(x / self.temp)
    #     return output

    def forward(self, x):
        x = Variable(x.view(-1, self.x_size))
        x = self.input_fc(x)
        x = F.relu(x)
        x = self.hidden_fc(x)
        x = F.relu(x)
        x = self.linear(x)
        output = torch.sigmoid(x / self.temp)
        return output

    # flip the reward function for action a, i.e., R(x,a) -> 1 - R(x,a) for all x
    def flip(self, a):
        with torch.no_grad():
            self.linear.weight[a] *= -1
            self.linear.bias[a] *= -1

    # select greedy action
    def act(self, rwd, random=False):
        if random:
            a = torch.randint(0, self.a_size, [1])
        else:
            a = torch.argmax(rwd)
        return a.item()

    def predict(self, x):
        x = Variable(x.view(-1, self.x_size))
        x = self.input_fc(x)
        x = F.relu(x)
        x = self.hidden_fc(x)
        x = F.relu(x)
        x = self.linear(x)
        output = F.log_softmax(x, dim=1)
        return output

    # def softmax_policy(self, x):
    #     x = Variable(x.view(-1, self.x_size))
    #     x = self.linear(x)
    #     output = F.softmax(x / self.temp, dim=1)
    #     return output

    def init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
            elif isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)


class decoder_Net(nn.Module):
    def __init__(self, y_size, a_size):
        super(decoder_Net, self).__init__()
        self.y_size = y_size
        self.temp = 1
        # self.linear = nn.Linear(y_size, a_size)
        self.input_fc = nn.Linear(y_size, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.linear = nn.Linear(100, a_size)

    # def forward(self, y):
    #     y = Variable(y.view(-1, self.y_size))
    #     y = self.linear(y)
    #     output = torch.sigmoid(y / self.temp)
    #     return output

    def forward(self, y):
        y = Variable(y.view(-1, self.y_size))
        y = self.input_fc(y)
        y = F.relu(y)
        y = self.hidden_fc(y)
        y = F.relu(y)
        y = self.linear(y)
        output = torch.sigmoid(y / self.temp)
        return output

    def flip(self, a):
        with torch.no_grad():
            self.linear.weight[a] *= -1
            self.linear.bias[a] *= -1

    def init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.zeros_(layer.weight)
            elif isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)