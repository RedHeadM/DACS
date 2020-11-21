import json

from data.base import *
from data.cityscapes_loader import cityscapesLoader
from data.gta5_dataset import GTA5DataSet
from data.synthia_dataset import SynthiaDataSet
from data.multiview import MulitviewSegLoader


def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        "cityscapes": cityscapesLoader,
        "gta": GTA5DataSet,
        "synthia": SynthiaDataSet,
        "multiview":MulitviewSegLoader
    }[name]

def get_data_path(name):
    """get_data_path
    :param name:
    :param config_file:
    """
    if name == 'cityscapes':
        return '../data/CityScapes/'
    if name == 'gta' or name == 'gtaUniform':
        return '../data/gta/'
    if name == 'synthia':
        return '../data/RAND_CITYSCAPES'
    if name == 'multiview':
        return '/tmp/texture_multibot_push_left10030/videos/train'
