import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from urbansurge.fault_diagnosis.data_utils import ANNDataset

class ANNClassifier(nn.Module):
    def __init__(self, input_features, output_features, lr, num_epochs, criterion,
                 shuffle_train=True, shuffle_test=False):
        super(ANNClassifier, self).__init__()

        # Training parameters.
        self.lr = lr # Learning rate.
        self.num_epochs = num_epochs # Number of epochs.
        self.criterion = criterion # Training criterion.

        # Training and testing shuffle.
        self.shuffle_train = shuffle_train
        self.shuffle_test = shuffle_test

        # Define model structure.
        # Fully connected layers
        self.fc1 = nn.Linear(input_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, output_features)  # output layer

        # Activation functions.
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Forward pass.
        x = x.view(x.size(0), -1)  # flatten the input, though if your input is already flat, this line is unnecessary
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.softmax(self.fc5(x))

        return x

    def fit_model(self, X_train, y_train):
        # Data loader.
        train_loader = self.prepare_data(X_train, y_train, self.shuffle_train)

        # Optimizer.
        optimizer = optim.Adam(self.parameters(), lr=self.lr)  # Adam optimizer

        for epoch in range(self.num_epochs):
            self.train()  # Set the model to training mode
            print('Epoch:', epoch)

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self(inputs)

                # Compute the loss
                loss = self.criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

        print('Training finished.')
        return self

    def test_model(self, X_test, y_test):
        # Test loader.
        test_loader = self.prepare_data(X_test, y_test, self.shuffle_test)

        # Storage for predictions and true labels
        all_predictions = []
        all_labels = []

        # Disable gradient calculations for efficiency
        with torch.no_grad():
            for inputs, labels in test_loader:
                # Forward pass: compute the model output
                outputs = self(inputs)
                predicted = outputs

                # Store predictions and labels
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Convert lists to arrays if necessary
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)

        return predictions, labels

    def predict(self, X):
        # Convert to torch tensor.
        X = torch.tensor(X, dtype=torch.float32)

        # Forward pass.
        outputs = self(X)

        return outputs

    @staticmethod
    def prepare_data(X, y, shuffle):
        # Convert X to torch tensor.
        X_prep = torch.tensor(X, dtype=torch.float32)

        # Convert y to categorical one-hot encoded tensor.
        y_prep = torch.tensor(y, dtype=torch.long)
        y_prep = torch.nn.functional.one_hot(y_prep, num_classes=torch.max(y_prep) + 1).type(torch.FloatTensor)

        # Create dataset and data loader.
        dataset = ANNDataset(X_prep, y_prep)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=shuffle)

        return dataloader