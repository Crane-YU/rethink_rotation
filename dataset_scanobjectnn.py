import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from utils.utils_data import normalize_pc, random_scale_pc, translate_pc, so3_rotate, z_rotate


def load_data(isTrain):
    all_data = []
    all_label = []
    partition = 'train' if isTrain else 'test'
    h5_name = 'data/h5_files/main_split/' + partition + '_objectdataset.h5'
    f = h5py.File(h5_name, 'r')
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    all_data.append(data)
    all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


class ScanObjectNN(Dataset):
    def __init__(self, train, num_points, normalize=False, transforms=False, rotate='so3', angle=None):
        self.data, self.label = load_data(train)
        self.num_points = num_points
        self.train = train
        self.transforms = transforms
        self.normalize = normalize
        self.rotate = rotate
        self.angle = angle

    def __getitem__(self, idx):
        current_points = self.data[idx]
        choice = np.random.choice(current_points.shape[0], self.num_points, replace=False)
        current_points = current_points[choice]
        label = self.label[idx]

        if self.normalize:
            current_points = normalize_pc(current_points)

        if self.train:
            if self.rotate == 'so3':
                current_points[:, :3] = so3_rotate(current_points[:, :3], self.angle)
            elif self.rotate == 'z':
                current_points[:, :3] = z_rotate(current_points[:, :3], self.angle)

            if self.transforms:
                current_points = random_scale_pc(current_points)
                # current_points = translate_pc(current_points)
        else:
            if self.rotate == 'so3':
                current_points[:, :3] = so3_rotate(current_points[:, :3], self.angle)
            elif self.rotate == 'z':
                current_points[:, :3] = z_rotate(current_points[:, :3], self.angle)

            if self.transforms:
                current_points = random_scale_pc(current_points)
                # current_points = translate_pc(current_points)

        return current_points, label

    def __len__(self):
        return self.data.shape[0]
