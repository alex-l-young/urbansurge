import torch.nn as nn


class SimpleANN(nn.Module):
    def __init__(self):
        super(SimpleANN, self).__init__()

        # Fully connected layers
        self.fc1 = nn.Linear(720, 512)  # input layer, 672 comes from 7x96
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)  # output layer

        # Activation function
        self.relu = nn.ReLU()


    def forward(self, x):
        # Forward pass.
        x = x.view(x.size(0), -1)  # flatten the input, though if your input is already flat, this line is unnecessary
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x) # No activation for regression.

        return x
