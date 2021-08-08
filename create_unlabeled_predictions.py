# encoding: utf-8

"""
The main CheXNet model implementation.
"""


import os
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
import progressbar

from utils.model import DenseNet121
import time

CKPT_PATH = './runs/effusion_train05%_teacher_00/densenet_model_weights_epoch14.pth'
LOGDIR = './runs/effusion_train05%_teacher_00'
N_CLASSES = 1
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
DATA_DIR = '/media/roberto/external/Data/images/images'
TRAIN_PSEUDOLABELS = './labels/binary_Effusion/train95%.txt'
TRAIN_FINETUNE = './labels/binary_Effusion/train05%.txt'

BATCH_SIZE = 16



def main():

    cudnn.benchmark = True

    model = DenseNet121(N_CLASSES).cuda()
    if os.path.isfile(CKPT_PATH):
        print("=> loading checkpoint")
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint)
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")
        raise

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    train_pseudo_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list_file=TRAIN_PSEUDOLABELS,
                                     transform=transforms.Compose([
                                         transforms.Resize(256),
                                         transforms.TenCrop(224),
                                         transforms.Lambda
                                         (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                         transforms.Lambda
                                         (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]),
                                     return_image_name=True)
    train_pseudo_loader = DataLoader(dataset=train_pseudo_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=8, pin_memory=True)

    train_finetune_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                       image_list_file=TRAIN_FINETUNE,
                                       transform=transforms.Compose([
                                           transforms.Resize(256),
                                           transforms.TenCrop(224),
                                           transforms.Lambda
                                           (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                           transforms.Lambda
                                           (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                       ]),
                                       return_image_name=True)
    train_finetune_loader = DataLoader(dataset=train_finetune_dataset, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=8, pin_memory=True)

    start_time = time.time()
    model.train()

    with torch.no_grad():

        ## MAKE LABELS FOR X% SPLIT
        predicted_data = []
        for i, (img_name, inp, target) in progressbar.progressbar(enumerate(train_pseudo_loader), max_value=train_pseudo_loader.__len__()):
            bs, n_crops, c, h, w = inp.size()
            input_var = torch.autograd.Variable(inp.view(-1, c, h, w).cuda(), volatile=True)
            output = model(input_var)
            output_mean = output.view(bs, n_crops, -1).mean(1)

            for pair in zip(img_name, output_mean, target):
                name = pair[0].split("/")[-1]
                pred = ' '.join([str(p) for p in pair[1].tolist()])
                labeled_data = f"{name} {pred} "
                predicted_data.append(labeled_data)
        total_time = time.time() - start_time
        print(f"Total time for labeling : {total_time}")
        # with open('./densenet_supervised10%_teacher/pseudo_labels90.json', 'w') as outfile:
        #     json.dump({"pseudo_labels":predicted_data}, outfile, indent=4)
        with open(f"{LOGDIR}/pseudo_labels95.txt", 'w') as outfile:
            for pred in predicted_data:
                outfile.write(pred+'\n')

        ## MAKE LABELS FOR Y% SPLIT
        predicted_data = []
        for i, (img_name, inp, target) in progressbar.progressbar(enumerate(train_finetune_loader), max_value=train_finetune_loader.__len__()):
            bs, n_crops, c, h, w = inp.size()
            input_var = torch.autograd.Variable(inp.view(-1, c, h, w).cuda(), volatile=True)
            output = model(input_var)
            output_mean = output.view(bs, n_crops, -1).mean(1)

            for pair in zip(img_name, output_mean, target):
                name = pair[0].split("/")[-1]
                pred = ' '.join([str(p) for p in pair[1].tolist()])
                labeled_data = f"{name} {pred} "
                predicted_data.append(labeled_data)
        total_time = time.time() - start_time
        print(f"Total time for labeling : {total_time}")
        # with open('./densenet_supervised10%_teacher/pseudo_labels10.json', 'w') as outfile:
        #     json.dump({"pseudo_labels":predicted_data}, outfile, indent=4)
        with open(f"{LOGDIR}/pseudo_labels05.txt", 'w') as outfile:
            for pred in predicted_data:
                outfile.write(pred+'\n')



if __name__ == '__main__':
    main()
