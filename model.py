from torch import nn

class VPPM(nn.Module):
    def __init__(self, n_features=21):
        super(VPPM, self).__init__()
        self.n_features = n_features

        self.linear = nn.Linear(self.n_features, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        return x