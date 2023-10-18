import numpy as np


class ModuleInfo(object):
    def __init__(self):
        pass

    @staticmethod
    def create_dataset(group, key, data, compression=False):
        if data is not np.nan and data.shape is not None:
            if compression: group.create_dataset(key, data=data, compression='gzip')
            else: group.create_dataset(key, data=data)
        else: group.create_dataset(key, dtype=np.uint8)

    @staticmethod
    def update_dataset(group, key, data, compression=False):
        if key in group.keys(): del group[key]
        ModuleInfo.create_dataset(group, key, data, compression)
