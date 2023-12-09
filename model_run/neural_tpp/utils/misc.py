# Copyright (c) Ant Group and its affiliates.

import glob
import os
import pickle

import numpy as np
import yaml


def py_assert(condition, exception_type, msg):
    """
    An assert function that ensures the condition holds, otherwise throws a message
    Args:
        condition: a formula of validity
        exception_type: Error type, such as ValueError
        msg: str, a message to throw out

    Returns:
        assertion result

    """
    if not condition:
        raise exception_type(msg)


def make_config_string(config, max_num_key=4):
    """

    Generate a name for config files
    Args:
        config: dict of config
        max_num_key: max number of keys to concat in the output

    Returns:
        a concatenated string from config dict
    """
    str_config = ''
    num_key = 0
    for k, v in config.items():
        if num_key < max_num_key:  # for the moment we only record model name
            if k == 'name':
                str_config += str(v) + '_'
                num_key += 1
    return str_config[:-1]


def save_config(save_dir, config):
    """

    General save config function, that save a dict object to yaml format file.
    Args:
        save_dir: str
            The path to save config file.
        config: dict
            The target config object.

    Returns:

    """
    prt_dir = os.path.dirname(save_dir)

    if not os.path.exists(prt_dir):
        os.makedirs(prt_dir)

    with open(save_dir, 'w') as f:
        yaml.dump(config, stream=f, default_flow_style=False, sort_keys=False)

    return


def load_config(config_dir, experiment_id='NHP'):
    """
    Args:
        config_dir: dir of yaml config file
        experiment_id: the customized name of the experiment, e.g., 'NHP_test'

    Returns:
        a dict of config

    """
    config = dict()
    model_configs = glob.glob(os.path.join(config_dir, 'model_config.yaml'))
    if not model_configs:
        model_configs = glob.glob(os.path.join(config_dir, 'model_config/*.yaml'))
    if not model_configs:
        raise RuntimeError('config_dir={} is not valid!'.format(config_dir))
    found_params = dict()

    for config_ in model_configs:
        with open(config_, 'r') as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            if experiment_id in config_dict:
                found_params[experiment_id] = config_dict[experiment_id]
        if len(found_params) == 2:
            break
    if experiment_id not in found_params:
        raise ValueError("expid={} not found in model config".format(experiment_id))
    # Update base settings first so that values can be overrided when conflict
    # with experiment_id settings
    found_params['model_id'] = experiment_id
    config.update(found_params)
    # Add data config
    data_config = load_dataset_config(config_dir, config[experiment_id]['model']['dataset_id'])
    config.update({'data': data_config})
    return config


def load_dataset_config(config_dir, dataset_id):
    dataset_configs = glob.glob(os.path.join(config_dir, 'dataset_config.yaml'))
    if not dataset_configs:
        dataset_configs = glob.glob(os.path.join(config_dir, 'dataset_config/*.yaml'))
    for config in dataset_configs:
        with open(config, 'r') as cfg:
            config_dict = yaml.load(cfg, Loader=yaml.FullLoader)
            if dataset_id in config_dict:
                return config_dict[dataset_id]
    raise RuntimeError('dataset_id={} is not found in config.'.format(dataset_id))


def create_folder(*args):
    """

    Create path if the folder doesn't exist.
    Args:
        *args: folder path that is to be created

    Returns:
        The folder's path.

    """

    path = os.path.join(*args)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def load_pickle(file_dir):
    """

    Args:
        file_dir: pickle folder path that is to be loaded

    Returns:
        data loaded from pickle file

    """
    try:
        data = pickle.load(file_dir, encoding='latin-1')
    except Exception:
        data = pickle.load(file_dir)

    return data


def has_key(target_dict, target_keys):
    if not isinstance(target_keys, list):
        target_keys = [target_keys]
    for k in target_keys:
        if k not in target_dict:
            return False
    return True


def array_pad_cols(arr, max_num_cols, pad_index):
    """

    Args:
        arr: original array
        max_num_cols: target num cols for padded array
        pad_index: pad index to fill out the padded elements

    Returns:
        padded array

    """
    res = np.ones((arr.shape[0], max_num_cols)) * pad_index

    res[:, :arr.shape[1]] = arr

    return res
