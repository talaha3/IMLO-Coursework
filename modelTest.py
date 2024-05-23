# Importing required packages
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


if __name__ == '__main__':
    # Mean and Standard Deviation
    mean = [0.4319, 0.3926, 0.3274]
    std = [0.3181, 0.2624, 0.3108]

    # Downloading test dataset and applying defined transformations
    testTransform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(torch.Tensor(mean), torch.Tensor(std)),
        ])

    testData = datasets.Flowers102(root='data', split = 'test', download=True, transform=testTransform)

    # Define model
    class MyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.convStack = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(32, 64, 3, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(64, 128, 3, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),

                nn.Conv2d(128, 128, 3, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(128, 256, 3, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(256, 256, 3, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(256, 512, 3, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),



            )

            self.Flatten = nn.Flatten()
            self.LinearStack = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(512, 2048),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(2048, 102),
            )

        def forward(self, x):
            x = self.convStack(x)
            x = self.Flatten(x)
            x = self.LinearStack(x)
            return x

    # Hyperparameters
    BATCH_SIZE = 32

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataloaders
    testLoader = DataLoader(testData, batch_size = BATCH_SIZE, shuffle = False)

    # Model
    model = MyCNN()
    params = torch.load("bestmodel.pt", map_location = device)
    model.load_state_dict(params)
    model = model.to(device)

    # Loss Function
    loss = nn.CrossEntropyLoss()

    # Defining function to calculate test accuracy
    def testModel(dataloader, model, lossFunction):
        model.eval()
        total = 0
        correct = 0
        epochLoss = 0

        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                total += y.size(0)
                epochLoss += lossFunction(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        epochLoss = epochLoss/total
        correct = (correct/total) * 100

        print(f"Test: Accuracy: {(correct):>0.1f}%, Avg loss: {epochLoss:>8f} \n")

    # Calculate test accuracy
    testModel(testLoader, model, loss)