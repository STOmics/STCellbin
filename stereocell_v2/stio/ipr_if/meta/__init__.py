import numpy as np
from h5py import File


class ModuleInfo(object):
    def __init__(self):
        pass

    def deserialize_to_h5(self, obj, group):
        dct = obj.__dict__.copy()
        for attr, v in dct.items():
            if "ClarityScores" in attr:
                print(attr)
            try:
                if isinstance(v, np.ndarray):
                    group.create_dataset(attr, data=v)
                elif isinstance(v, list):
                    group.create_dataset(attr, data=np.array(v))
                elif isinstance(v, dict):
                    if attr not in group.keys():
                        group.create_group(attr)
                    group_sub = group[attr]
                    # for row_column, points in clarity_scores.items():
                    #     clarity.create_dataset(row_column, data=points)
                    for key, value in v.items():
                        group_sub.create_dataset(key, data=np.array(value))
                elif isinstance(v, ModuleInfo):
                    info_group = group.create_group(attr, track_order=True)
                    self.deserialize_to_h5(v, info_group)
                else:
                    if v is None:
                        v = '-'
                    group.attrs[attr] = v
            except:
                print("{}:{} is error".format(attr, v))

    def to_file(self, mIF_class_dict:dict, file_path:str):
        with File(file_path, 'w') as f:
            for key, value in mIF_class_dict.items():
                group_sub = f.create_group(key, track_order=True)
                self.deserialize_to_h5(value, group_sub)

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


