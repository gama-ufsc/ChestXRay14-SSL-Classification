import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet

import progressbar

from utils.model import DenseNet121
import time

N_CLASSES = 1
DATA_DIR = '/media/roberto/external/Data/images/images'
BATCH_SIZE = 16


def make_predictions(model_path,
                     labels_file,
                     output_file):
    """
    Create predictions with the teacher model
    Results in a single file wiht prediction for the trained class (only one label)
    Args:
        model_path:  model's weights
        labels_file: annotations for the dataset
        output_file: where to store the predictions

    """
    cudnn.benchmark = True

    model = DenseNet121(N_CLASSES).cuda()
    if os.path.isfile(model_path):
        print("=> loading checkpoint")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")
        raise

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    dataset_gen = ChestXrayDataSet(data_dir=DATA_DIR,
                                   image_list_file=labels_file,
                                   transform=transforms.Compose([
                                       transforms.Resize(256),
                                       transforms.TenCrop(224),
                                       transforms.Lambda
                                       (lambda crops: torch.stack(
                                           [transforms.ToTensor()(crop) for crop in crops])),
                                       transforms.Lambda
                                       (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                   ]),
                                   return_image_name=True)
    dataset_loader = DataLoader(dataset=dataset_gen, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=8, pin_memory=True)

    start_time = time.time()
    model.train()

    with torch.no_grad():

        ## MAKE LABELS FOR X% SPLIT
        predicted_data = []
        for i, (img_name, inp, target) in progressbar.progressbar(enumerate(dataset_loader),
                                                                  max_value=dataset_loader.__len__()):
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
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as outfile:
            for pred in predicted_data:
                outfile.write(pred + '\n')


def filter_predictions(predictions_file,
                       labels_file,
                       output_file,
                       data_class=2,
                       k_best=0.5):
    """
    We filter the predictions from the teacher model, based on the score and class balance of data
    Currently only filtering a single class (data_class)
    Args:
        predictions_file: txt file with the teacher's predictions
        labels_file:  labels file for the same images
        output_file:  where the filtered prediction will be stored
        data_class:  what label is being used
        k_best:      fraction of predictions with highest and lowest scores to be used in the pseudolabels generation
    """
    # Open predictions
    with open(predictions_file, 'r') as file:
        data_preds = file.read()
        data_preds = data_preds.split('\n')

        images_names_pseudo = [entry.split(' ')[0] for entry in data_preds if len(entry) > 10]
        preds = [[float(p) for p in entry.split(' ')[1:-1]] for entry in data_preds if len(entry) > 10]
        preds = np.array(preds)

    # Open labels
    with open(labels_file, 'r') as file:
        data_labels = file.read()
        data_labels = data_labels.split('\n')
        images_names_finetune = [entry.split(' ')[0] for entry in data_labels]
        labels = [[int(p) for p in entry.split(' ')[1:]] for entry in data_labels if len(entry) > 10]
        labels = np.array(labels)

    # Based on the chosen class (data_class)
    # We discover the class ratio in the dataset
    effusion_labels = labels[:, data_class]
    effusion_ratio = np.sum(effusion_labels) / effusion_labels.shape[0]
    print(f"The ratio betwen label 1 (class) and label 0 (normal) is {effusion_ratio}")

    # Create a empty prediction table
    # -1 means ignore entry
    filtered_preds = -1 * np.ones([preds.shape[0], 14])  # Start ignoring all entries

    # Filter the predictions based on the class balance
    # and a constant K for chossing the K% best predictions
    k_high_predictions = int(k_best * effusion_ratio * preds.shape[0])
    k_low_predictions = int(k_best * (1 - effusion_ratio) * preds.shape[0])

    print(f"From a total of {preds.shape[0]} predicitons, we are taking : ")
    print(
        f" -- Getting {k_high_predictions}   [{round(100 * k_high_predictions / preds.shape[0], 2)}% of total] high predictions")
    print(
        f" -- Getting {k_low_predictions}  [{round(100 * k_low_predictions / preds.shape[0], 2)}% of total]low predictions")

    # We get the probabilities bounds for filtering predictions
    # We will use only predictions with p>upper_bound or p<lower_bound
    n_preds = preds[:, 0]
    n_argsorted = np.argsort(n_preds)
    # the upper bound of probability to be included in the ssl training
    k_highprob = n_preds[n_argsorted[-k_high_predictions]]
    # the lower bound of probability to be included in the ssl training
    k_lowprob = n_preds[n_argsorted[k_low_predictions]]
    print(f"The probability limits are:")
    print(f" -- {k_highprob} for label 1")
    print(f" -- {k_lowprob} for label 0")

    filtered_preds[preds[:, 0] >= k_highprob, 2] = 1
    filtered_preds[preds[:, 0] <= k_lowprob, 2] = 0

    # Save the data
    data = ''
    for image, filtered_pred in zip(images_names_pseudo, filtered_preds.astype(np.int8)):
        if filtered_pred[2] == -1:
            continue
        data += image + ' ' + ' '.join([str(p) for p in filtered_pred])
        data += '\n'

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as file:
        file.write(data)
