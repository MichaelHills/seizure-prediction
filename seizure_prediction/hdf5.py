import h5py
import numpy as np
import re
from common.data import jsdict

# Helper method to dump a dictionary of ndarrays or primitives to hdf5, and then read them back.
# It looks like I also added list support, cool.

METADATA_TAG = '__metadata'

list_regex = re.compile(r"""__list_(.*)_(\d+)""")


def write(filename, obj):
    data = h5py.File(filename, 'w-', libver='latest')
    meta_dataset = data.create_dataset(METADATA_TAG, shape=(1,))

    for key in obj.keys():
        value = obj[key]
        if isinstance(value, np.ndarray):
            data.create_dataset(key, data=value)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                assert isinstance(v, np.ndarray)
                data.create_dataset('__list_%s_%d' % (key, i), data=v)
        else:
            meta_dataset.attrs[key] = value

    data.close()


def read(filename):
    data = h5py.File(filename, 'r')
    obj = {}
    for key in data.keys():
        value = data[key]
        if key == METADATA_TAG:
            for metakey in value.attrs.keys():
                obj[metakey] = value.attrs[metakey]
        elif not key.startswith('__list'):
            obj[key] = value[:]

    list_keys = [key for key in data.keys() if key.startswith('__list')]
    if len(list_keys) > 0:
        list_keys.sort()
        for key in list_keys:
            match = list_regex.match(key)
            assert match is not None
            list_key = match.group(1)
            list_index = int(match.group(2))
            out_list = obj.setdefault(list_key, [])
            assert len(out_list) == list_index
            out_list.append(data[key][:])

    data.close()

    return jsdict(obj)

