import os
import numpy as np
import pandas as pd
from glob import glob

from tqdm import tqdm

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os, random

torch.set_default_dtype(torch.float32)

info = pd.read_csv('data_info.csv')
root_dir = '/dartfs-hpc/rc/home/q/f004kkq/COSC189/data/'
fname_pattern = 'sub{:03d}_run{:03d}.npy'
roi = 'HG'

random.seed(42)
np.random.seed(42)

def data_loader(batch_size, roi, mode, leave_out, shuffle):
    dataset = TemporalData(root_dir, roi, mode, leave_out)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

def load_data(data_path, mode, leave_out):
    labels = []
    data = []

    for sub_id in range(1, 21):
        for run_id in range(1, 9):
            tmp_label = info.loc[(info[leave_out] == mode) & (info['sub'] == sub_id) & (info['run'] == run_id), 'label']
            if not len(tmp_label):
                continue
            labels.append(tmp_label.to_list())
            tmp_data = np.load(os.path.join(data_path, 'sub{:03d}_run{:03d}.npy'.format(sub_id, run_id)))
            data.append(tmp_data)
    data = np.concatenate(data)
    data = np.expand_dims(data, axis=1).astype(np.float32)
    
    labels = np.concatenate(labels)
#     labels = np.expand_dims(labels, axis=1)
    
    return data, labels.astype(np.int)

class TemporalData(Dataset):
    def __init__(self, root_dir, roi='HG', mode='train', leave_out='run_out', shuffle=True):
        '''
        :param data_path: path for saving pre-processed data, train, eval, test
        :return:
        '''
        data_path = os.path.join(root_dir, roi)
        self.data, self.labels = load_data(data_path, mode, leave_out)
        
        if shuffle:
            index = list(range(len(self.labels)))
            random.shuffle(index)
            self.data = self.data[index]
            self.labels = self.labels[index]
            
        self.transform = False

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]