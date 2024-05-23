# Importing required packages
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt 


#----DATA AUGMENTATION----

# Augmentation hyperparameters
ROTATION = 25
BRIGHTNESS = 0.15
CONTRAST = 0.5
SATURATION = 0.5

# Mean and Standard Deviation
# Values are calculated using mean, std of images from training set (see dataset_mean_std.ipynb)
mean = [0.4319, 0.3926, 0.3274]
std = [0.3181, 0.2624, 0.3108]

# Define individual and combo training transformation for data augmentation to artificially increase the training dataset size. 
# Including changing BCS, random rotation, random horizontal flips
# and resize image to (256,256), crop to (224,224) and normalisation by given mean and std
brightTransform= transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ColorJitter(BRIGHTNESS, 0, 0, 0),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std)),
    ])

contrastTransform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ColorJitter(0, CONTRAST, 0, 0),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std)),
    ])

saturateTransform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ColorJitter(0, 0, SATURATION, 0),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std)),
    ])

bcTransform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ColorJitter(BRIGHTNESS, CONTRAST, 0, 0),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std)),
    ])

csTransform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ColorJitter(0, CONTRAST, SATURATION, 0),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std)),
    ])

bsTransform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ColorJitter(BRIGHTNESS, 0, SATURATION, 0),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std)),
    ])

cropTransform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std)),
    ])

comboTransform1 = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ColorJitter(BRIGHTNESS, CONTRAST, SATURATION, 0),
    transforms.RandomRotation(ROTATION),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std)),
    ])

comboTransform2 = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ColorJitter(BRIGHTNESS, 0, 0, 0),
    transforms.RandomRotation(ROTATION),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std)),
    ])

comboTransform3 = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ColorJitter(0, CONTRAST, 0, 0),
    transforms.RandomRotation(ROTATION),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std)),
    ])

comboTransform4 = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ColorJitter(0, 0, SATURATION, 0),
    transforms.RandomRotation(ROTATION),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std)),
    ])

comboTransform5 = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ColorJitter(BRIGHTNESS, 0, SATURATION, 0),
    transforms.RandomRotation(25),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std)),
    ])

comboTransform6 = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ColorJitter(BRIGHTNESS, CONTRAST, 0, 0),
    transforms.RandomRotation(ROTATION),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std)),
    ])

comboTransform7 = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ColorJitter(0, CONTRAST, SATURATION, 0),
    transforms.RandomRotation(ROTATION),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std)),
    ])

# Creating multiple datasets using each transform type to artificially increase dataset size
brightData1 = datasets.Flowers102(root='data', split = 'train', download=True, transform=brightTransform)
brightData2 = datasets.Flowers102(root='data', split = 'train', download=True, transform=brightTransform)
brightData3 = datasets.Flowers102(root='data', split = 'train', download=True, transform=brightTransform)

contrastData1 = datasets.Flowers102(root='data', split = 'train', download=True, transform=contrastTransform)
contrastData2 = datasets.Flowers102(root='data', split = 'train', download=True, transform=contrastTransform)
contrastData3 = datasets.Flowers102(root='data', split = 'train', download=True, transform=contrastTransform)

saturateData1 = datasets.Flowers102(root='data', split = 'train', download=True, transform=saturateTransform)
saturateData2 = datasets.Flowers102(root='data', split = 'train', download=True, transform=saturateTransform)
saturateData3 = datasets.Flowers102(root='data', split = 'train', download=True, transform=saturateTransform)

bcData1 = datasets.Flowers102(root='data', split = 'train', download=True, transform=bcTransform)
bcData2 = datasets.Flowers102(root='data', split = 'train', download=True, transform=bcTransform)
bcData3 = datasets.Flowers102(root='data', split = 'train', download=True, transform=bcTransform)

csData1 = datasets.Flowers102(root='data', split = 'train', download=True, transform=csTransform)
csData2 = datasets.Flowers102(root='data', split = 'train', download=True, transform=csTransform)
csData3 = datasets.Flowers102(root='data', split = 'train', download=True, transform=csTransform)

bsData1 = datasets.Flowers102(root='data', split = 'train', download=True, transform=bsTransform)
bsData2 = datasets.Flowers102(root='data', split = 'train', download=True, transform=bsTransform)
bsData3 = datasets.Flowers102(root='data', split = 'train', download=True, transform=bsTransform)

cropData1 = datasets.Flowers102(root='data', split = 'train', download=True, transform=cropTransform)
cropData2 = datasets.Flowers102(root='data', split = 'train', download=True, transform=cropTransform)

combo1Data1 = datasets.Flowers102(root='data', split = 'train', download=True, transform=comboTransform1)
combo1Data2 = datasets.Flowers102(root='data', split = 'train', download=True, transform=comboTransform1)
combo1Data3 = datasets.Flowers102(root='data', split = 'train', download=True, transform=comboTransform1)
combo1Data4 = datasets.Flowers102(root='data', split = 'train', download=True, transform=comboTransform1)

combo2Data1 = datasets.Flowers102(root='data', split = 'train', download=True, transform=comboTransform2)
combo2Data2 = datasets.Flowers102(root='data', split = 'train', download=True, transform=comboTransform2)
combo2Data3 = datasets.Flowers102(root='data', split = 'train', download=True, transform=comboTransform2)
combo2Data4 = datasets.Flowers102(root='data', split = 'train', download=True, transform=comboTransform2)

combo3Data1 = datasets.Flowers102(root='data', split = 'train', download=True, transform=comboTransform3)
combo3Data2 = datasets.Flowers102(root='data', split = 'train', download=True, transform=comboTransform3)
combo3Data3 = datasets.Flowers102(root='data', split = 'train', download=True, transform=comboTransform3)
combo3Data4 = datasets.Flowers102(root='data', split = 'train', download=True, transform=comboTransform3)

combo4Data1 = datasets.Flowers102(root='data', split = 'train', download=True, transform=comboTransform4)
combo4Data2 = datasets.Flowers102(root='data', split = 'train', download=True, transform=comboTransform4)
combo4Data3 = datasets.Flowers102(root='data', split = 'train', download=True, transform=comboTransform4)
combo4Data4 = datasets.Flowers102(root='data', split = 'train', download=True, transform=comboTransform4)

combo5Data1 = datasets.Flowers102(root='data', split = 'train', download=True, transform=comboTransform5)
combo5Data2 = datasets.Flowers102(root='data', split = 'train', download=True, transform=comboTransform5)
combo5Data3 = datasets.Flowers102(root='data', split = 'train', download=True, transform=comboTransform5)
combo5Data4 = datasets.Flowers102(root='data', split = 'train', download=True, transform=comboTransform5)

combo6Data1 = datasets.Flowers102(root='data', split = 'train', download=True, transform=comboTransform6)
combo6Data2 = datasets.Flowers102(root='data', split = 'train', download=True, transform=comboTransform6)
combo6Data3 = datasets.Flowers102(root='data', split = 'train', download=True, transform=comboTransform6)
combo6Data4 = datasets.Flowers102(root='data', split = 'train', download=True, transform=comboTransform6)

combo7Data1 = datasets.Flowers102(root='data', split = 'train', download=True, transform=comboTransform7)
combo7Data2 = datasets.Flowers102(root='data', split = 'train', download=True, transform=comboTransform7)
combo7Data3 = datasets.Flowers102(root='data', split = 'train', download=True, transform=comboTransform7)
combo7Data4 = datasets.Flowers102(root='data', split = 'train', download=True, transform=comboTransform7)

# Concatenating different transformed datasets into one
trainData = ConcatDataset([brightData1,
                          brightData2,
                          brightData3,

                          contrastData1,
                          contrastData2,
                          contrastData3,

                          saturateData1,
                          saturateData2,
                          saturateData3,

                          bcData1,
                          bcData2,
                          bcData3,

                          csData1,
                          csData2,
                          csData3,

                          bsData1,
                          bsData2,
                          bsData3,

                          cropData1,
                          cropData2,

                          combo1Data1,
                          combo1Data2,
                          combo1Data3,
                          combo1Data4,

                          combo2Data1,
                          combo2Data2,
                          combo2Data3,
                          combo2Data4,

                          combo3Data1,
                          combo3Data2,
                          combo3Data3,
                          combo3Data4,

                          combo4Data1,
                          combo4Data2,
                          combo4Data3,
                          combo4Data4,

                          combo5Data1,
                          combo5Data2,
                          combo5Data3,
                          combo5Data4,

                          combo6Data1,
                          combo6Data2,
                          combo6Data3,
                          combo6Data4,

                          combo7Data1,
                          combo7Data2,
                          combo7Data3,
                          combo7Data4])

# Downloading datasets by pre-defined splits and applying defined transformations
valTransform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std)),
    ])

testTransform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std)),
    ])

valData = datasets.Flowers102(root='data', split = 'val', download=True, transform=valTransform)
testData = datasets.Flowers102(root='data', split = 'test', download=True, transform=testTransform)

# Displaying a few images from the training dataset
figure = plt.figure(figsize=(10, 10))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(trainData), size=(1,)).item()
    img, label = trainData[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.axis("off")
    plt.imshow(img.permute(1,2,0), cmap="gray")
plt.show()


#----MODEL----

# Creating CNN for image classification
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
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 102),
        )

    def forward(self, x):
        x = self.convStack(x)
        x = self.Flatten(x)
        x = self.LinearStack(x)
        return x

# Model hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 50
LR = 0.0001
GAMMA = 0.9

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Dataloaders
trainLoader = DataLoader(trainData, batch_size = BATCH_SIZE, shuffle = True)
valLoader = DataLoader(valData, batch_size = BATCH_SIZE, shuffle = False)
testLoader = DataLoader(testData, batch_size = BATCH_SIZE, shuffle = False)

# Model
model = MyCNN()
model = model.to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA, last_epoch=-1)

# Loss Function
loss = nn.CrossEntropyLoss()


#----TRAINING----
# Defining function for model training
losses = []
accuracy = []
def trainModel(dataloader, model, lossFunction, optimizer):
    model.train()
    currentLoss = 0.0
    correct = 0
    total = 0
    epochLoss = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        total += y.size(0)

        optimizer.zero_grad()
        pred = model(X)
        loss = lossFunction(pred, y)
        loss.backward()
        optimizer.step()

        currentLoss += loss.item()
        epochLoss += lossFunction(pred, y).item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    epochLoss = epochLoss/total
    correct = correct/total

    losses.append(epochLoss)
    accuracy.append(correct * 100)
    print(f'Training: Accuracy {correct * 100:>0.1f}%, Loss: {currentLoss / len(dataloader):.5f}, Epoch Loss: {epochLoss:.5f}')

# Defining function to calculate validation accuracy
def validateModel(dataloader, model, lossFunction):
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

    print(f"Validation: Accuracy: {(correct):>0.1f}%, Avg loss: {epochLoss:>8f} \n")
    return epochLoss, correct


# Applying training function and output accuracy at each epoch iteration
bestValAccuracy = 0.0
bestValLoss = float('inf')
bestEpoch = 0

valLosses = []
valAccuracies = []

for epoch in range(NUM_EPOCHS):
    print(f'Epoch {epoch+1}:')
    trainModel(trainLoader, model, loss, optimizer)
    valLoss, valAccuracy = validateModel(valLoader, model, loss)
    valLosses.append(valLoss)
    valAccuracies.append(valAccuracy)
    scheduler.step()

    if (valAccuracy > bestValAccuracy):
        torch.save(model.state_dict(), 'bestmodel.pt')
        bestValLoss = valLoss
        bestValAccuracy = valAccuracy
        bestEpoch = epoch+1

print(f'Best Accuracy: {bestValAccuracy}. Best Loss: {bestValLoss} Best Epoch: {bestEpoch}')
    