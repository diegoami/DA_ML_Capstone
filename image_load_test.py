import torch
import torchvision
import os
import torchdata as td
root = 'wendy_cnn_frames_data'
total_count = sum([len(files) for r, d, files in os.walk(root)])
BATCH_SIZE=64
NUM_WORKER=1
from torchvision import transforms
data_transform = torchvision.transforms.Compose(
    [
        transforms.ToTensor()
    ]
)

# Single change, makes an instance of torchdata.Dataset
# Works just like PyTorch's torch.utils.data.Dataset, but has
# additional capabilities like .map, cache etc., see project's description
model_dataset = td.datasets.WrapDataset(torchvision.datasets.ImageFolder(root, transform=data_transform))
# Also you shouldn't use transforms here but below
train_count = int(0.7 * total_count)
valid_count = int(0.2 * total_count)
test_count = total_count - train_count - valid_count
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    model_dataset, (train_count, valid_count, test_count)
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
test_dataset_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER
)
dataloaders = {
    "train": train_dataset_loader,
    "val": valid_dataset_loader,
    "test": test_dataset_loader,
}
for batch_ndx, sample in enumerate(train_dataset_loader):
    print(batch_ndx)
    print(sample)