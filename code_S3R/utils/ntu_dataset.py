import glob
import torch
import numpy
from torch.utils import data


class NTURGBDDataset(data.Dataset):
    
    # Homepage: http://rose1.ntu.edu.sg/datasets/actionrecognition.asp

    def __init__(self, split, set_name, data_type='pose_3d', dataset_root_folder='/.../my_pose_datasets/my_ntu/'):
        assert data_type in ['pose_3d']  # TODO: add the rest
        assert split in ['subject', 'view']
        assert set_name in ['train', 'test']
        self.meta_data = {
            # subject = person
            'train_subject': [ 1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38],
            'test_subject': [3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 36, 37, 39, 40],
            # view = camera
            'train_view': [2, 3],
            'test_view': [1]
        }
        self.set_name = set_name
        self.split = split
        self.data_type = data_type
        self._x_fns = []
        self._ys = []
        for fn in sorted(glob.glob(dataset_root_folder + 'data_as_torch_tensors/data_pose3d__*__skeleton_*.pt')):
            _S, _C, _P, _R, _A = self._parse_ntu_rgbd_filename(fn)
            if self.split == 'view' and _C in self.meta_data['{}_view'.format(set_name)]:
                self._x_fns.append(fn)
                self._ys.append(_A)
            if self.split == 'subject' and _P in self.meta_data['{}_subject'.format(set_name)]:
                self._x_fns.append(fn)
                self._ys.append(_A)
    
    def __str__(self):
        return 'NTU Dataset (data_type={}, split={}, set_name={})'.format(self.data_type, self.split, self.set_name)
    
    def __repr__(self):
        return self.__str__()
            
    def _parse_ntu_rgbd_filename(self, filename):
        short_fn = filename.split('/')[-1].split('.')[0].split('__')[1]
        S = int(short_fn[1:4])    # from 1 to 17  --> S = ...
        C = int(short_fn[5:8])    # from 1 to 3  --> C = camera_id
        P = int(short_fn[9:12])   # from 1 to 40 --> P = subject_id
        R = int(short_fn[13:16])  # from 1 to 2  --> R = ...
        A = int(short_fn[17:20])  # from 1 to 60 --> A = action_class
        return S, C, P, R, A
    
    def __len__(self):
        return len(self._x_fns)

    def __getitem__(self, index):
        x = torch.load(self._x_fns[index])
        y = self._ys[index]
        return x, y

