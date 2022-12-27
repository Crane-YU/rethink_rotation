import os
import numpy as np
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset
from utils.utils_data import normalize_pc, random_scale_pc, translate_pc, so3_rotate, z_rotate


class ModelNet40(Dataset):
    def __init__(self, train, use_normal=False, normalize=False, transforms=False, rotate='so3', angle=None):
        super().__init__()
        self.train = train
        self.transforms = transforms
        self.normalize = normalize
        self.rotate = rotate
        self.angle = angle

        if train:
            self.points = np.load('data/ModelNet40_normal_1024_train_points.npy')
            self.labels = np.load('data/ModelNet40_normal_1024_train_label.npy')
        else:
            self.points = np.load('data/ModelNet40_normal_1024_test_points.npy')
            self.labels = np.load('data/ModelNet40_normal_1024_test_label.npy')

        if not use_normal:
            self.points = self.points[:, :, :3]

        print('Successfully load ModelNet40 with', self.points.shape[0], 'instances')

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])  # 1024

        if self.train:
            np.random.shuffle(pt_idxs)

        current_points = self.points[idx, pt_idxs].copy()
        if self.normalize:
            current_points[:, :3] = normalize_pc(current_points[:, :3])

        if self.train:
            if self.rotate == 'so3':
                current_points[:, :3] = so3_rotate(current_points[:, :3], self.angle)
            elif self.rotate == 'z':
                current_points[:, :3] = z_rotate(current_points[:, :3], self.angle)

            if self.transforms:
                current_points[:, :3] = random_scale_pc(current_points[:, :3])
                current_points[:, :3] = translate_pc(current_points[:, :3])
        else:
            if self.rotate == 'so3':
                current_points[:, :3] = so3_rotate(current_points[:, :3], self.angle)
            elif self.rotate == 'z':
                current_points[:, :3] = z_rotate(current_points[:, :3], self.angle)

            if self.transforms:
                current_points[:, :3] = random_scale_pc(current_points[:, :3])
                current_points[:, :3] = translate_pc(current_points[:, :3])

        normal = current_points[:, :3]
        label = self.labels[idx]
        return current_points, normal, label

    def __len__(self):
        return self.points.shape[0]
