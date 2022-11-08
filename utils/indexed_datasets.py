import pickle
from copy import deepcopy
import numpy as np


class IndexedDataset:
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.data_file = None
        index_data = np.load(f"{path}.idx", allow_pickle=True).item()
        self.byte_offsets = index_data['offsets']
        self.id2pos = index_data.get('id2pos', {})
        self.data_file = open(f"{path}.data", 'rb', buffering=-1)

    def check_index(self, i):
        if i < 0 or i >= len(self.byte_offsets) - 1:
            raise IndexError('index out of range')

    def __del__(self):
        if self.data_file:
            self.data_file.close()

    def __getitem__(self, i):
        if self.id2pos is not None and len(self.id2pos) > 0:
            i = self.id2pos[i]
        self.check_index(i)
        self.data_file.seek(self.byte_offsets[i])
        b = self.data_file.read(self.byte_offsets[i + 1] - self.byte_offsets[i])
        item = pickle.loads(b)
        return item

    def __len__(self):
        return len(self.byte_offsets) - 1

    def __iter__(self):
        self.iter_i = 0
        return self

    def __next__(self):
        if self.iter_i == len(self):
            raise StopIteration
        else:
            item = self[self.iter_i]
            self.iter_i += 1
            return item


class IndexedDatasetBuilder:
    def __init__(self, path, append=False):
        self.path = path
        if append:
            self.data_file = open(f"{path}.data", 'ab')
            index_data = np.load(f"{path}.idx", allow_pickle=True).item()
            self.byte_offsets = index_data['offsets']
            self.id2pos = index_data.get('id2pos', {})
        else:
            self.data_file = open(f"{path}.data", 'wb')
            self.byte_offsets = [0]
            self.id2pos = {}

    def add_item(self, item, id=None):
        s = pickle.dumps(item)
        bytes = self.data_file.write(s)
        if id is not None:
            self.id2pos[id] = len(self.byte_offsets) - 1
        self.byte_offsets.append(self.byte_offsets[-1] + bytes)

    def finalize(self):
        self.data_file.close()
        np.save(open(f"{self.path}.idx", 'wb'),
                {'offsets': self.byte_offsets, 'id2pos': self.id2pos})
