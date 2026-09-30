import copy
import yaml
import json
import pickle
import random
import numpy as np

import torch


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def pickle_data(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def read_config(path):
    with open(path, "r") as f:
        return copy.deepcopy(yaml.load(f, Loader=yaml.Loader))


def write_json(data, path):
    with open(path, "w", encoding="UTF-8") as f:
        json.dump(data, f, indent=4)
