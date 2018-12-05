import torch

from code_S3R.my_utils import other_utils as utils


class OnlineDHGDataset(torch.utils.data.Dataset):
    def __init__(self, set='train'):
        x_train, y_train = utils.load_dataset_in_torch(use_online_dataset=True, use_14=True,
                                                       train_dataset=True)
        x_test, y_test = utils.load_dataset_in_torch(use_online_dataset=True, use_14=True,
                                                     train_dataset=False)
        y_train = y_train.type_as(torch.FloatTensor())
        y_test = y_test.type_as(torch.FloatTensor())
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.set = set
        self.split_train_validation = 0.7
        self.limit_split = int(self.split_train_validation * len(self.x_train))

    def __len__(self):
        if self.set == 'train':
            return len(self.x_train[:self.limit_split])
        elif self.set == 'validation':
            return len(self.x_train[self.limit_split:])
        else:  # test
            return len(self.x_test)

    def __getitem__(self, index):
        if self.set == 'train':
            return self.x_train[index], self.y_train[index]
        elif self.set == 'validation':
            return self.x_train[self.limit_split - 1 + index], self.y_train[self.limit_split - 1 + index]
        else:  # test
            return self.x_test[index], self.y_test[index]


