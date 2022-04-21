import os
import shutil
import json
import torch
import yaml
import pickle
import time

import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def get_file_name():
    return time.asctime()+'_checkpoint.pth.tar'

def store_data(model, optimizer, scheduler, hyperparams):
    '''
    Saves checkpoint: model, optimizer, scheduler and hyperparameters
    '''
    checkpoint = {}
    checkpoint['model'] = model.state_dict()
    checkpoint['optimizer'] = optimizer.state_dict()
    checkpoint['scheduelr'] = scheduler.state_dict()
    checkpoint['hyperparams'] = hyperparams
    save_checkpoint(checkpoint, filename = get_file_name())

def load_data(model, optimizer, scheduler, filename):
    '''
    Loads checkpoint: model, optimizer, scheduler and hyperparameters
    '''
    checkpoint = torch.load(filename)
    model.load_state(checkpoint['model'])
    optimizer.load_state(checkpoint['optimizer'])
    scheduler.load_state(checkpoint['scheduler'])
    return checkpoint['hyperparams']


def save_dict_to_pickle(state, basepath = os.getcwd(), filename='config.p'):
    '''
    Saves dictionary into pickle format
    Code from: https://stackoverflow.com/questions/7100125/storing-python-dictionaries
    '''
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    savepath = os.path.join(basepath, filename)
    with open(savepath, 'wb') as fp:
        pickle.dump(state, fp, protocol=pickle.HIGHEST_PROTOCOL)


def save_dict_to_json(state, basepath = os.getcwd(), filename='config.json'):
    '''
    Saves dictionary into json format
    '''
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    savepath = os.path.join(basepath, filename)
    with open(savepath, 'w') as fp:
        json.dump(state, fp)

def save_checkpoint(state, basepath = os.getcwd(), filename='checkpoint.pt'):
    '''
    Saves model state dict
    '''
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    torch.save(state, os.path.join(basepath, filename))


def save_config_file(model_checkpoints_folder, args):
    '''
    Saves checkpoints as yaml file
    '''
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


# Extract csv data from a Tensorboard directory
def tflog2pandas(path):

    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data
