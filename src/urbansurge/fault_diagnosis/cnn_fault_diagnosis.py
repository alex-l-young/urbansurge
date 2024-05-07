# CNN to identify all conduit diameters.

# Library imports.
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Activation and pooling
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers for regression
        self.fc1 = nn.Linear(64 * 1 * 24, 128)  # after 3 max-poolings, size becomes 1x24
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)  # for the 5x1 output

    def forward(self, x):
        # Apply first convolutional layer, then activation, then pooling
        x = self.maxpool(self.relu(self.conv1(x)))

        # Apply second convolutional layer, then activation, then pooling
        x = self.maxpool(self.relu(self.conv2(x)))

        # Apply third convolutional layer, then activation, then pooling
        # x = self.maxpool(self.relu(self.conv3(x)))
        x = self.relu(self.conv3(x))

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        # No activation at the last layer as it's regression
        x = self.fc3(x)

        return x

if __name__ == '__main__':
    # Create the model instance
    model = SimpleCNN()
    print(model)
