"""
This script trains a model for the Kaggle Digit Recognizer challenge and saves the checkpoints.
"""

import logging
import numpy as np

from tqdm import tqdm
from copy import deepcopy

from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import f1_score, confusion_matrix

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss, Linear, Conv2d
from torch.nn import Sequential, ReLU, MaxPool2d, Dropout2d, Flatten, Dropout, LogSoftmax
from torch.optim import SGD, lr_scheduler

from torchvision.models.squeezenet import squeezenet1_1


class Digits(Dataset):
    def __init__(self, path):
        """Load the dataset on init"""

        # Dataset is relatively small, so it makes sense to simply load in into memory as a gigantic numpy array.
        self.data = np.genfromtxt(path, delimiter=',', skip_header=1, max_rows=42000)
        self.labels = None

        # Check data shape
        if self.data.shape[1] > 785:
            raise ValueError
        # Check if the data has labels by the number of columns then separate it
        elif self.data.shape[1] == 785:
            self.labels = self.data[:, 0]
            self.data = self.data[:, 1:]

        # Sanity check
        logging.info(f'Dataset items: {len(self)}')

        # Check class ratio
        if len(self.labels > 0):
            unique, counts = np.unique(self.labels, return_counts=True)
            logging.info(counts)

    def normalise(self, mean=None, std=None):
        # Calculate the normalisation parameters that hasn't been specified
        _mean = self.data.mean() if not mean else mean
        _std = self.data.std() if not std else std
        self.data = (self.data - _mean) / _std

        return mean, std

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        """Return a row reshaped to a 28x28 image and its corresponding label"""

        mat = self.data[item, :].reshape((28, 28))
        # Torch Conv2D layers require 3D image representations with channels in the first dimension
        mat = np.expand_dims(mat, axis=0)

        img = torch.tensor(mat, dtype=torch.float)

        if len(self.labels):
            label = self.labels[item].astype(int)
            return img, label
        else:
            return img, None


def get_model(model, device):
    if model == 'squeezenet':
        # Grab a pretrained Squeezenet from torchvision and send to GPU
        model = squeezenet1_1(pretrained=True).to(device)
        # Modify the first convolutional layer to expect greyscale images
        model.features[0] = Conv2d(in_channels=1,
                                   out_channels=model.features[0].out_channels,
                                   kernel_size=model.features[0].kernel_size,
                                   stride=model.features[0].stride).to(device)
        # Modify the final convolutional layer in the classifier to produce only 10 channels that correspond to the
        # number of classes in this classification problem.
        model.classifier[1] = Conv2d(in_channels=model.classifier[1].in_channels,
                                     out_channels=10,
                                     kernel_size=model.classifier[1].kernel_size,
                                     stride=model.classifier[1].stride).to(device)

        return model.to(device)

    elif model == 'simple_cnn':
        # This is a sequential implementation of a common cnn architecture for MNIST
        model = Sequential(Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
                           ReLU(inplace=True),
                           Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
                           ReLU(inplace=True),
                           MaxPool2d(kernel_size=(2, 2)),
                           Dropout2d(p=0.25, inplace=False),
                           Flatten(),
                           Linear(in_features=12*12*64, out_features=128),
                           ReLU(inplace=True),
                           Dropout(p=0.5, inplace=False),
                           Linear(in_features=128, out_features=10),
                           LogSoftmax(dim=1))

        return model.to(device)


def run_epoch(dataloader, model, criterion, device,
              optimizer=None, is_training=False, calculate_metrics=False):
    """Run a single epoch for training or cross validation"""

    epoch_loss = 0

    # Empty arrays to track metrics
    prediction_all = np.array([], dtype=np.uint8).reshape(0, 1)
    target_all = np.array([], dtype=np.uint8).reshape(0, 1)

    for batch in dataloader:
        if is_training:
            # Don't accumulate gradients over batches.
            # Not required for testing as it's wrapped in a no_grad context
            optimizer.zero_grad()

        # Grab a batch and send it to GPU
        data, target = batch

        data, target = data.to(device), target.to(device)

        # Forward pass
        output = model(data)
        loss = criterion(output.squeeze(), target)
        epoch_loss += loss.item()

        if calculate_metrics:
            prediction = output.squeeze().argmax(dim=1, keepdim=True)
            prediction_all = np.vstack([prediction_all, prediction.cpu().numpy()])
            target_all = np.vstack([target_all, target.view_as(prediction).cpu().numpy()])

        if is_training:
            # Backward pass only during training
            loss.backward()
            optimizer.step()

    if calculate_metrics:
        metrics = {'f1': f1_score(target_all.flatten(), prediction_all.flatten(), average=None)}
        cm = confusion_matrix(target_all.flatten(), prediction_all.flatten())

        return epoch_loss, metrics, cm

    return epoch_loss


def train(path):

    # Session config
    device = torch.device("cuda:0")

    # Load dataset
    dataset = Digits(path=path)

    # Split the dataset
    rs = ShuffleSplit(n_splits=1, train_size=0.7)
    split = rs.split(X=dataset.data, y=dataset.labels)
    train_idx, test_idx = next(split)
    train_set, test_set = dataset, deepcopy(dataset)
    train_set.data = train_set.data[train_idx, :]
    train_set.labels = train_set.labels[train_idx]
    test_set.data = test_set.data[test_idx, :]
    test_set.labels = test_set.labels[test_idx]

    # Normalise the train set
    mean, std = train_set.normalise()
    # Apply the same normalisation parameters to the test set
    test_set.normalise(mean=mean, std=std)

    # Define data loaders
    train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=512, pin_memory=True)
    test_loader = DataLoader(dataset=test_set, shuffle=False, batch_size=512, pin_memory=True)

    model = 'squeezenet'
    #model = 'simple_cnn'
    model = get_model(model=model, device=device)

    # Simple CrossEntropyLoss for multi-class classification with single-digit encoding
    criterion = CrossEntropyLoss()
    lr, gamma = 0.01, 0.99
    optimizer = SGD(lr=lr, momentum=0.0, params=model.parameters())
    # Exponentially decrease learning rate
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=gamma)

    # Put model in train mode
    model.train()

    for epoch in tqdm(range(1000)):
        epoch_loss = run_epoch(dataloader=train_loader, model=model, criterion=criterion, optimizer=optimizer,
                               device=device, is_training=True, calculate_metrics=False)

        logging.info(f' Train epoch loss: {epoch_loss}, LR: {lr}')

        # Step the scheduler for each epoch
        scheduler.step()
        lr *= gamma  # Has no effect except for printing the LR

        # Run a cross validation check for each n-th training epoch
        if epoch and not epoch % 50:
            # Disable dropout for testing
            model.eval()
            # Don't need to track gradients for XV
            with torch.no_grad():
                epoch_loss, metrics, cm = run_epoch(dataloader=test_loader, model=model,
                                                    criterion=criterion, device=device,
                                                    is_training=False, calculate_metrics=True)

                logging.info('Confusion matrix:')
                logging.info(cm)
                logging.info(f'Epoch {epoch:04d} F1 Scores:')
                logging.info('\n'.join(f'{cls}: {f1}' for cls, f1 in zip(range(10), metrics['f1'])))

            # Save checkpoint
            torch.save(model.state_dict(), f'ckp{epoch:04d}.pt')

            # Re-enable dropout layers
            model.train()


if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    logging.getLogger('matplotlib.font_manager').disabled = True

    train(path='/home/attila/Datasets/digit-recognizer/train.csv')
