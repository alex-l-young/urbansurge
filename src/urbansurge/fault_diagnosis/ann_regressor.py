import torch
from torch.utils.data import DataLoader

from urbansurge.fault_diagnosis.ann_classifier import ANNClassifier
from urbansurge.fault_diagnosis.data_utils import ANNDataset

class ANNRegressor(ANNClassifier):
    def __init__(self, input_features, output_features, lr, num_epochs, criterion,
                 shuffle_train=True, shuffle_test=False):
        super(ANNRegressor, self).__init__(input_features, output_features, lr, num_epochs, criterion,
                 shuffle_train=shuffle_train, shuffle_test=shuffle_test)

    def forward(self, x):
        # Forward pass.
        x = x.view(x.size(0), -1)  # flatten the input, though if your input is already flat, this line is unnecessary
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x) # No activation for regression.

        return x

    @staticmethod
    def prepare_data(X, y, shuffle):
        # Convert X to torch tensor.
        X_prep = torch.tensor(X, dtype=torch.float32)

        # Convert y to categorical one-hot encoded tensor.
        y_prep = torch.tensor(y, dtype=torch.float32)
        y_prep = y_prep.unsqueeze(1)

        # Create dataset and data loader.
        dataset = ANNDataset(X_prep, y_prep)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=shuffle)

        return dataloader
