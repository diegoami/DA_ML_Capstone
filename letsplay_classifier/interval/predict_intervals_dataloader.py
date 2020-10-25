from predict import model_fn, get_model_info
import argparse
import os
import json
import torch
from constants import IMG_WIDTH, IMG_HEIGHT
import torchdata as td
import torchvision
from torchvision import transforms


from interval.predict_intervals_utils import convert_to_intervals, get_short_classes

if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change

    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    parser.add_argument('--img-width', type=int, default=IMG_WIDTH, metavar='N',
                        help='width of image (default: 128)')
    parser.add_argument('--img-height', type=int, default=IMG_HEIGHT, metavar='N',
                        help='height of image (default: 72)')
    parser.add_argument('--layer-cfg', type=str, default='D', metavar='N',
                        help='layer type for VGG')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')

    args = parser.parse_args()

    model = model_fn(args.model_dir)
    model_info = get_model_info(args.model_dir)
    classes = model_info['class_names']
    short_classes = get_short_classes(classes)

    total_count = sum([len(files) for r, d, files in os.walk(args.data_dir)])

    data_transform = torchvision.transforms.Compose(
        [
            transforms.Resize((args.img_height, args.img_width)),
            transforms.ToTensor()
        ]
    )

    # build a dataset of images from the img_dir directory

    im_folder = torchvision.datasets.ImageFolder(args.data_dir, transform=data_transform)
    model_dataset = td.datasets.WrapDataset(im_folder)
    dataset_loader = torch.utils.data.DataLoader(model_dataset)

    model.train(False)
    model.eval()

    ev_seqs = []
    second_tot = 0
    for i, data in enumerate(dataset_loader):
        inputs, labels = data
        outputs = model(inputs.cuda() if use_gpu else inputs)
        out_logs = torch.exp(outputs).detach().cpu().numpy()
        for out_log in out_logs:
            out_log = (out_log / sum(out_log) * 20) .astype(int)
            categor_str = ''.join([min(int(out_log[x]), 20) * short_classes[x] for x in range(0, 5)]).rjust(20,'_')
            ev_seqs.append(categor_str)

    convert_to_intervals(ev_seqs, short_classes, classes)