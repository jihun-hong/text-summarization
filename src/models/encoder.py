import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h)
        return sent_scores


class DNNEncoder(nn.Module):
    def __init__(self, hidden_size, num_units=128, num_layers=2):
        super(DNNEncoder, self).__init__()
        self.num_layers = num_layers
        self.linear1 = nn.Linear(hidden_size, num_units)
        self.actives = nn.ModuleList([nn.LeakyReLU() for _ in range(num_layers + 1)])
        self.linears = nn.ModuleList([nn.Linear(num_units, num_units) for _ in range(num_layers)])
        self.linearf = nn.Linear(num_units, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = self.linear1(x)
        for i in range(self.num_layers):
            h = self.linears[i](self.actives[i](h))
        h = self.linearf(self.actives[self.num_layers](h)).squeeze(-1)

        sent_scores = self.sigmoid(h)
        return sent_scores
