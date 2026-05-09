import os
import json
import time
import utils
import models
import logger
import inspect
import datetime
import argparse
import data_loader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np

parser = argparse.ArgumentParser()
# basic args
parser.add_argument('--task', type = str)
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--epochs', type = int, default = 100)

# evaluation args
parser.add_argument('--weight_file', type = str)
parser.add_argument('--result_file', type = str)

# cnn args
parser.add_argument('--kernel_size', type = int)

# rnn args
parser.add_argument('--pooling_method', type = str)

# multi-task args
parser.add_argument('--alpha', type = float)

# log file name
parser.add_argument('--log_file', type = str)

args = parser.parse_args()

config = json.load(open('./config.json', 'r'))

def train(model, elogger, train_set, eval_set):
    # record the experiment setting
    elogger.log(str(model))
    elogger.log(str(args._get_kwargs()))

    model.train()

    if torch.cuda.is_available():
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr = 1e-4)

    for epoch in range(args.epochs):
        model.train()
        print ("Training on epoch {}".format(epoch))
        for input_file in train_set:
            print ("Train on file {}".format(input_file))

            # data loader, return two dictionaries, attr and traj
            data_iter = data_loader.get_loader(input_file, args.batch_size)

            running_loss = 0.0

            for idx, (attr, traj) in enumerate(data_iter):
                # transform the input to pytorch variable
                attr, traj = utils.to_var(attr), utils.to_var(traj)

                _, loss = model.eval_on_batch(attr, traj, config)

                # update the model
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                running_loss += loss.item()
                print('\r Progress {:.2f}%, average loss {}'.format(
                    (idx + 1) * 100.0 / len(data_iter),
                    running_loss / (idx + 1.0)
                ), end='')
            elogger.log('Training Epoch {}, File {}, Loss {}'.format(epoch, input_file, running_loss / (idx + 1.0)))

        # evaluate the model after each epoch
        evaluate(model, elogger, eval_set, save_result = False)

        # save the weight file after each epoch
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        weight_name = f"weight_{timestamp}"
        elogger.log('Save weight file {}'.format(weight_name))
        torch.save(model.state_dict(), './saved_weights/' + weight_name)

def write_result(fs, pred_dict, attr, input_file, row_start):
    pred = pred_dict['pred'].data.cpu().numpy()
    label = pred_dict['label'].data.cpu().numpy()

    if row_start == 0:
        fs.write('source_file\trow_id\tdriverID\tdateID\ttimeID\tlabel\tpred\n')

    for i in range(pred_dict['pred'].size()[0]):
        dateID = attr['dateID'].data[i].item()
        timeID = attr['timeID'].data[i].item()
        driverID = attr['driverID'].data[i].item()

        fs.write('%s\t%d\t%d\t%d\t%d\t%.6f\t%.6f\n' % (
            input_file,
            row_start + i,
            driverID,
            dateID,
            timeID,
            label[i][0],
            pred[i][0]
        ))

    return row_start + pred_dict['pred'].size()[0]


def evaluate(model, elogger, files, save_result = False):
    model.eval()
    if save_result:
        fs = open('%s' % args.result_file, 'w')

    row_id = 0

    for input_file in files:
        running_loss = 0.0
        data_iter = data_loader.get_loader(input_file, args.batch_size)

        for idx, (attr, traj) in enumerate(data_iter):
            attr, traj = utils.to_var(attr), utils.to_var(traj)

            pred_dict, loss = model.eval_on_batch(attr, traj, config)

            if save_result:
                row_id = write_result(fs, pred_dict, attr, input_file, row_id)

            running_loss += loss.item()

        print('Evaluate on file {}, loss {}'.format(input_file, running_loss / (idx + 1.0)))
        elogger.log('Evaluate File {}, Loss {}'.format(input_file, running_loss / (idx + 1.0)))

    if save_result: fs.close()

def get_kwargs(model_class):
    model_args = inspect.getfullargspec(model_class.__init__).args
    shell_args = args._get_kwargs()

    kwargs = dict(shell_args)

    for arg, val in shell_args:
        if not arg in model_args or val is None:
            kwargs.pop(arg)

    return kwargs

def run():
    # get the model arguments
    kwargs = get_kwargs(models.DeepTTE.Net)

    # model instance
    model = models.DeepTTE.Net(**kwargs)

    # experiment logger
    elogger = logger.Logger(args.log_file)

    if args.task == 'train':
        train(model, elogger, train_set = config['train_set'], eval_set = config['eval_set'])

    elif args.task == 'test':
        # load the saved weight file
        model.load_state_dict(torch.load(args.weight_file))
        if torch.cuda.is_available():
            model.cuda()
        evaluate(model, elogger, config['test_set'], save_result = True)

if __name__ == '__main__':
    run()
