import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import training dataset and resize to (224,224)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
trainData = datasets.Flowers102(root='data', split = 'train', download=True, transform=transform)
trainLoader = DataLoader(trainData, batch_size = 10, shuffle = False)

# Calculates mean and standard deviation of the dataset
def mean_std(loader):
  mean = 0
  std = 0
  image_no = 0

  for images, _ in loader:
    image_count_batch = images.size(0)
    images = images.view(image_count_batch, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    image_no += image_count_batch

  mean /= image_no
  std /= image_no

  return mean, std

mean, std = mean_std(trainLoader)
print(mean, std)