from predict import model_fn, predict_fn, input_fn, output_fn
import argparse
import os
import json
import torch
import numpy as np
from constants import IMG_WIDTH, IMG_HEIGHT
import torchdata as td
import torchvision
from torchvision import transforms
from torch.autograd import Variable

so = 'BH_ST'
ls = ['Battle', 'Hideout', 'Other', 'Siege', 'Tournament']
def remove_outliers(seqs):
    mseq = []
    def_value = '_'*20
    for i, x in enumerate(seqs):
        if i == 0:
            pred_x = def_value
        else:
            pred_x = seqs[i-1]
        if i == len(seqs)-1:
            seq_x = def_value
        else:
            seq_x = seqs[i+1]
        curr_x = x
        if ((curr_x.count('_') > 15) and (pred_x.count('_') < 5) and(seq_x.count('_') < 5)):
            curr_x = ''.join([(pred_x+seq_x).count(so[x])*so[x] for x in range(0,5)])
        if ((curr_x.count('_') < 15) and (pred_x.count('_') > 5) and(seq_x.count('_') > 5)):
            curr_x = ''.join([((pred_x+seq_x).count(so[x])//2)*so[x] for x in range(0,5)])
        mseq.append(curr_x)
    return mseq


def get_hour_format(second_tot):
    hour, minute, second = second_tot // 3600, (second_tot // 60) % 60, second_tot % 60
    time_tpl = map(str, (hour, minute, second)) if hour > 0 else map(str, (minute, second))
    current_time = ':'.join([x.zfill(2) for x in time_tpl])
    return current_time

if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    use_gpu = torch.cuda.is_available()
    # Here we set up an argument parser to easily access the parameters
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
            current_time = get_hour_format(second_tot)
            categor_str = ''.join([min(int(out_log[x]), 20)*so[x] for x in range(0,5)]).rjust(20,'_')
            ev_seqs.append(categor_str)
    one_n =  remove_outliers(ev_seqs)
    two_n = remove_outliers(one_n)
    time_seqs = []
    in_battle = False
    start_battle = None
    ev_battle_str = ''
    for i, (x, y, z) in enumerate(zip(ev_seqs, one_n, two_n)):
        current_time = get_hour_format(second_tot)
        print(f'{current_time} {x} {y} {z}')
        second_tot += 2
        if not in_battle:
            if z.count('_') < 8:
                start_battle = current_time
                in_battle = True
        else:
            if z.count('_') > 12:
                end_battle = current_time
                time_seqs.append((start_battle, end_battle, ev_battle_str))
                start_battle, end_battle = None, None
                in_battle = False
                ev_battle_str = ''
            else:
                ev_battle_str += z
    for start_battle, end_battle, ev_battle_str in time_seqs:
        prob_list = [int(ev_battle_str.count(so[x]) / len(ev_battle_str) * 100) for x in range(0,5)]
        prob_str = ', '.join([f'{ls[x]} : {prob_list[x]}% ' for x in range(0,5) if prob_list[x] > 5 ])
        print(f'{start_battle}-{end_battle} | {prob_str}')
