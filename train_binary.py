# encoding: utf-8

"""
The main CheXNet model implementation.
"""
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torch.optim as optim
import progressbar

from torch.utils.tensorboard import SummaryWriter

from utils.model import create_model
from utils.data_loader import train_data_loader, validation_data_loader
from utils.evaluation import evaluate_model

import time

cudnn.benchmark = True

from config import DATA_DIR, CLASS_NAMES, N_CLASSES


def train(CHOSEN_CLASS,
          CKPT_PATH,
          TRAIN_IMAGE_LIST,
          VAL_IMAGE_LIST,
          EPOCHS,
          LR,
          LR_STEP,
          RUN_NAME,
          LOADER_TYPE='original',
          TRAIN_BATCH_SIZE=14,
          VALIDATION_BATCH_SIZE=8
          ):
    logdir = RUN_NAME #f'./runs/{RUN_NAME}'
    writer = SummaryWriter(log_dir=logdir)
    model = create_model(CKPT_PATH).cuda()

    train_loader = train_data_loader(DATA_DIR, TRAIN_IMAGE_LIST, TRAIN_BATCH_SIZE, LOADER_TYPE)
    val_loader = validation_data_loader(DATA_DIR, VAL_IMAGE_LIST, VALIDATION_BATCH_SIZE)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=LR_STEP)
    for epoch in range(EPOCHS):

        # # skip first 15 epochs
        # if epoch <= 15:
        #     print(f'SKIPPING EPOCH {epoch}')
        #     for i in range(train_loader.__len__()):
        #         optimizer.step()
        #     scheduler.step()
        #     continue

        running_loss = 0.0
        start_time = time.time()
        model.train()
        for i, (inp, target) in progressbar.progressbar(enumerate(train_loader), max_value=train_loader.__len__()):
            target = target[:, CHOSEN_CLASS].reshape([-1, 1])
            target = target.cuda()

            if LOADER_TYPE == 'five_cropped':
                inp = inp.cuda()
                optimizer.zero_grad()
                bs, ncrops, c, h, w = inp.size()
                output = model(inp.view(-1, c, h, w))
                output = output.view(bs, ncrops, -1).mean(1)
            else: # original, strong, etc
                inp = inp.cuda()
                optimizer.zero_grad()
                output = model(inp)

            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
                writer.add_scalar('loss/train', running_loss / 100)
                writer.flush()
                running_loss = 0
        total_time = time.time() - start_time
        print(f"Total time for trainig on epoch {epoch + 1} : {total_time}")
        writer.add_scalar('lr/train', optimizer.param_groups[0]['lr'], epoch)
        evaluate_model(model, val_loader, CHOSEN_CLASS, total_time, epoch, writer, logdir)
        scheduler.step()

    writer.close()


def test(CHOSEN_CLASS,
          CKPT_PATH,
          TEST_IMAGE_LIST,
          RUN_NAME,
          TEST_BATCH_SIZE=8
          ):
    logdir = RUN_NAME  # f'./runs/{RUN_NAME}'
    writer = SummaryWriter(log_dir=logdir)
    model = create_model(CKPT_PATH).cuda()

    test_loader = validation_data_loader(DATA_DIR, TEST_IMAGE_LIST, TEST_BATCH_SIZE)

    start_time = time.time()
    evaluate_model(model, test_loader, CHOSEN_CLASS, 0, 0, writer, logdir)
    total_time = time.time() - start_time
    print(f"Total time for test on epoch : {total_time}")

    writer.close()
