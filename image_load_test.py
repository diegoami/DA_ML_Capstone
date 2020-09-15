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
num_epochs=10
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
#    print(sample)

model = net.CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

train_losses = []
valid_losses = []

#for epoch in range(MAX_EPOCHS):
    # Training
#    for local_batch, local_labels in train_dataset_loader:
        # Transfer to GPU
#        local_batch, local_labels = local_batch.to(device), local_labels.to(device)


for epoch in range(1, num_epochs + 1):
    # keep-track-of-training-and-validation-loss
    train_loss = 0.0
    valid_loss = 0.0

    # training-the-model
    model.train()
    for data, target in train_dataset_loader:
        # move-tensors-to-GPU
        data = data.to(device)
        target = target.to(device)

        # clear-the-gradients-of-all-optimized-variables
        optimizer.zero_grad()
        # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
        output = model(data)
        # calculate-the-batch-loss
        loss = criterion(output, target)
        # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
        loss.backward()
        # perform-a-ingle-optimization-step (parameter-update)
        optimizer.step()
        # update-training-loss
        train_loss += loss.item() * data.size(0)

    # validate-the-model
    model.eval()
    for data, target in valid_dataset_loader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)

        loss = criterion(output, target)

        # update-average-validation-loss
        valid_loss += loss.item() * data.size(0)

    # calculate-average-losses
    train_loss = train_loss / len(train_dataset_loader.sampler)
    valid_loss = valid_loss / len(valid_dataset_loader.sampler)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    # print-training/validation-statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))

print('Finished Training')
torch.save(net, 'model.pth')