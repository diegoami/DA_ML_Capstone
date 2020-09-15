import torch
import torchvision
import os
import torchdata as td
import torch.optim as optim
from ytframe import net
import torch.nn as nn

root = 'wendy_cnn_frames_data'
total_count = sum([len(files) for r, d, files in os.walk(root)])
BATCH_SIZE=64
NUM_WORKER=1
MAX_EPOCHS=10
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}
from torchvision import transforms
IMG_HEIGHT = 32
IMG_WIDTH = 32
data_transform = torchvision.transforms.Compose(
    [
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor()

    ]
)

# Single change, makes an instance of torchdata.Dataset
# Works just like PyTorch's torch.utils.data.Dataset, but has
# additional capabilities like .map, cache etc., see project's description
model_dataset = td.datasets.WrapDataset(torchvision.datasets.ImageFolder(root, transform=data_transform))
# Also you shouldn't use transforms here but below
train_count = int(0.7 * total_count)
valid_count = total_count - train_count

train_dataset, valid_dataset= torch.utils.data.random_split(
    model_dataset, (train_count, valid_count)
)

# Apply transformations here only for train dataset

#train_dataset = train_dataset.map(data_transform)

# Rest of the code goes the same

train_dataset_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER
)
valid_dataset_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER
)

dataloaders = {
    "train": train_dataset_loader,
    "val": valid_dataset_loader

}
#for batch_ndx, sample in enumerate(train_dataset_loader):
#    print(batch_ndx)
#       print(sample)

net = net.CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


#for epoch in range(MAX_EPOCHS):
    # Training
#    for local_batch, local_labels in train_dataset_loader:
        # Transfer to GPU
#        local_batch, local_labels = local_batch.to(device), local_labels.to(device)


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_dataset_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
 #       if i % 2000 == 1999:    # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' %
              (epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.0

print('Finished Training')