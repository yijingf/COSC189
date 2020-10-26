import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os, random

random.seed(42)
np.random.seed(42)


class SpatialDataset(Dataset):
    def __init__(self, root_path='./data/spatial', mode='train', max_interval_size=3, transform=None):
        '''
        :param root_path: path for saving pre-processed data, train, eval, test
        :return:
        '''
        self.dataset_path = os.path.join(root_path, mode)
        self.label2int = {'ambient':0, 'symphonic':1, 'metal':2, 'rocknroll':3, 'country':4}
        self.item_list = sorted(os.listdir(self.dataset_path))
        self.max_interval_size = max_interval_size
        self.transform = transform
        print(self.dataset_path, len(self.item_list))

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, index):
        item_name = self.item_list[index]
        data_label_dict = torch.load(os.path.join(self.dataset_path, item_name))
        data, label = data_label_dict['img'], self.label2int[data_label_dict['label']]

        # pad information or split, [z, time, x, y]
        time_len = data.shape[1]
        if time_len < self.max_interval_size:
            pad = np.zeros([data.shape[0], self.max_interval_size-time_len] + list(data.shape[2:]))
            data = np.concatenate([data, pad], axis=1)
        else:
            rand_idx = [i for i in range(time_len)]
            random.shuffle(rand_idx)
            rand_idx = rand_idx[:self.max_interval_size]
            data = data[:, rand_idx, :, :]
        data = data.astype(np.float32)
        if self.transform:
            data = torch.from_numpy(data).div(255)
        # data = data.transpose((2, 1, 0, 3)) # x channel
        # data = data.transpose((3, 1, 2, 0)) # y channel
        return data, np.array([label], dtype=np.int)


class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = targets.shape[0]
        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()
        return n_correct_elems / batch_size


if __name__ == '__main__':
    s_dataset = SpatialDataset(mode='train',transform=True)
    data_loader = DataLoader(s_dataset, batch_size=4, shuffle=True)
    for each in data_loader:
        imgs, labels = each
        print(imgs.shape, labels.shape)