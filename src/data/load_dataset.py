import feather
import pandas as pd
from pathlib import Path


def get_dataset_filename(config: dict, data_type: str, dataset_type: str):
    """
    data_type = {'meta', 'ts'}
    dataset_type = {'train', 'test'}
    """
    path = config['dataset']['input_directory']
    path += config['dataset']['files'][data_type][dataset_type]

    return Path(path)


def load_dataset(config, data_type: str, debug_mode: bool):
    """
    data_type = {'meta', 'ts'}
    """
    train_path = get_dataset_filename(config, data_type, 'train')
    test_path = get_dataset_filename(config, data_type, 'test')
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    return train, test


def save_dataset(train_path, test_path, train, test):
    feather.write_dataframe(train, train_path)
    feather.write_dataframe(test, test_path)
