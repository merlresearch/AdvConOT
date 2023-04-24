# Copyright (c) 2020,2022 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch
import torch.utils.data as data
import numpy as np
import pickle
from sklearn.preprocessing import normalize


def preprocess(x):
    x = normalize(x, axis=1)
    return x

def load_data_i3d(split_num = 1):
    #get_num_frames = lambda xx: [int(x[2]) for x in xx]
    num_frames = 0 # we do not use this field. If you want to use it, then you
    # will need to find the number of frames in the videos. This was used in the DSP algorithm, wang and cherian, ECCV 2018.
    data_path= './data/hmdb_icml_clean_i3d_split' + str(split_num) + '.pkl'
    # we only provide 10 data samples in ./data folder to have a sense of the data format the algorithm expects.
    # to create the full i3d data use the insructions above.

    print('loading %s'%(data_path))
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    num_data = len(data['flow_data'])
    rgb_data = []
    flow_data = []
    for t in range(num_data):
        rgb_data.append(preprocess(data['rgb_data'][t].transpose()))
        flow_data.append(preprocess(data['flow_data'][t].transpose()))

    labels = np.array(data['labels']) #[:,0].astype('int')
    train_idx = np.array(data['train_idx']) #[0].astype('int')-1
    test_idx = np.array(data['test_idx']) #[0].astype('int')-1
    return rgb_data, flow_data, labels, train_idx, test_idx, num_frames

def normalize_data(ff, mean, mmax):
    if len(ff.shape)==2:
        if ff.shape[0]>1:
            ff_mean = ff.mean(0)[np.newaxis,:]
        else:
            ff_mean = ff
    else:
        ff_mean = ff[np.newaxis,:]
    return ff, ff_mean

class Train_Set(data.Dataset):

    def __init__(self, data_type, split_num=1):
        self.rgb_data, self.flow_data, self.labels, self.train_idx, self.test_idx, self.num_frames = load_data_i3d(split_num)
        if data_type == 'rgb':
            self.dsize = self.rgb_data[0].shape[1]
            self.all_data = np.concatenate(self.rgb_data, axis=0)
            self.mean = self.all_data.mean(0)[:self.dsize][np.newaxis,:]
            self.max = self.all_data.max()
        elif data_type == 'flow':
            self.dsize = self.flow_data[0].shape[1]
            self.all_data = np.concatenate(self.flow_data, axis=0)
            self.mean = self.all_data.mean(0)[:self.dsize][np.newaxis,:]
            self.max = self.all_data.max()
        else:
            print('unknown datatype')

        self.data_type = data_type

    def __getitem__(self, index):
        idx = self.train_idx[index]
        if self.data_type == 'rgb':
            nf = self.rgb_data[idx].shape[0]

            if nf > 2:
                fidx = np.random.permutation(nf)[0]
            else:
                fidx = 0
            ff = np.array(self.rgb_data[idx])[fidx,:self.dsize]
            ff, ff_mean = normalize_data(ff, self.mean, self.max)
            return torch.tensor(ff_mean), self.labels[idx] #-1
        elif self.data_type == 'flow':
            nf = self.flow_data[idx].shape[0]
            if nf > 2:
                fidx = np.random.permutation(nf)[0]
            else:
                fidx = 0
            ff = np.array(self.flow_data[idx])[fidx,:self.dsize]
            ff, ff_mean = normalize_data(ff, self.mean, self.max)
            return torch.tensor(ff_mean), self.labels[idx] #-1
        else:
            print('unknown data type. it should be either rgb/flow for jhmdb dataset')

    def __len__(self):
        return len(self.train_idx)

class Test_Set(data.Dataset):
    def __init__(self, data_type, split_num=1):
        self.data_type = data_type
        self.rgb_data, self.flow_data, self.labels, self.train_idx, self.test_idx, self.num_frames = load_data_i3d(split_num)
        if data_type == 'rgb':
            self.dsize = self.rgb_data[0].shape[1]
            self.all_data = np.concatenate(self.rgb_data, axis=0)
            self.mean = self.all_data.mean(0)[:self.dsize][np.newaxis,:]
            self.max = self.all_data.max()
        elif data_type == 'flow':
            self.dsize = self.flow_data[0].shape[1]
            self.all_data = np.concatenate(self.flow_data, axis=0)
            self.mean = self.all_data.mean(0)[:self.dsize][np.newaxis,:]
            self.max = self.all_data.max()
        else:
            print('unknown datatype')

        self.data_type = data_type

    def __getitem__(self, index):
        idx = self.test_idx[index]
        if self.data_type == 'rgb':
            ff = np.array(self.rgb_data[idx])[:,:self.dsize]
            ff, ff_mean = normalize_data(ff, self.mean, self.max)
            return torch.tensor(ff), self.labels[idx] #-1
        elif self.data_type == 'flow':
            ff = np.array(self.flow_data[idx])[:,:self.dsize]
            ff, ff_mean = normalize_data(ff, self.mean, self.max)
            return torch.tensor(ff), self.labels[idx] #-1
        else:
            print('unknown data type. it should be either rgb/flow for jhmdb dataset')

    def __len__(self):
        return len(self.test_idx)


class All_Data_Set(data.Dataset):
    def __init__(self, data_type, split_num=1):
        self.rgb_data, self.flow_data, self.labels, self.train_idx, self.test_idx, self.num_frames = load_data_i3d(split_num)
        if data_type == 'rgb':
            self.dsize = self.rgb_data[0].shape[1]
            self.all_data = 0
            self.mean = 0
            self.max = 0
        elif data_type == 'flow':
            self.dsize = self.flow_data[0].shape[1]
            self.all_data = 0
            self.mean = 0
            self.max = 0
        else:
            print('unknown datatype')
        self.data_type = data_type

    def __getitem__(self, index):
        idx = index #self.alldata_idx[index]
        if self.data_type == 'rgb':
            ff = np.array(self.rgb_data[idx])[:,:self.dsize]
            ff, ff_mean = normalize_data(ff, self.mean, self.max)
            return (torch.tensor(ff), self.labels[idx], torch.tensor(ff_mean), index) # -1
        elif self.data_type == 'flow':
            ff = np.array(self.flow_data[idx])[:,:self.dsize]
            ff, ff_mean = normalize_data(ff, self.mean, self.max)
            return (torch.tensor(ff), self.labels[idx], torch.tensor(ff_mean), index) # -1
        else:
            print('unknown data type. it should be either rgb/flow for jhmdb dataset')

    def __len__(self):
        return len(self.labels)

def train_collate_fn(data):
    data, classids = zip(*data)
    return torch.cat(data, dim=0).float(), torch.tensor(classids)

def test_collate_fn(data):
    data, classids = zip(*data)
    return data, torch.tensor(classids)

def all_data_collate_fn(data):
    data, classids, data_mean, data_index = zip(*data)
    return data, classids, data_mean, data_index

# data_type is either flow or rgb.
def get_train_loader(data_type, batch_size=100, shuffle=True, num_workers=6, pin_memory=True, split_num=1):
    train_set = Train_Set(data_type, split_num)
    data_loader = torch.utils.data.DataLoader(dataset=train_set,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=train_collate_fn,
                                              pin_memory=pin_memory)
    return data_loader

def get_test_loader(data_type, batch_size=100, shuffle=False, num_workers=0, pin_memory=True, split_num=1):
    test_set = Test_Set(data_type, split_num)
    data_loader = torch.utils.data.DataLoader(dataset=test_set,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=test_collate_fn,
                                              pin_memory=pin_memory)
    return data_loader

# loads both train and test.
def get_all_data_loader(data_type, batch_size=100, shuffle=False, num_workers=0, pin_memory=True, split_num=1):
    all_data_set = All_Data_Set(data_type, split_num)
    data_loader = torch.utils.data.DataLoader(dataset=all_data_set,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=all_data_collate_fn,
                                              pin_memory=pin_memory)
    return data_loader
