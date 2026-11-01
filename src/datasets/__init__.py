from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, imgs, labels):
        self._data = imgs
        self._label = labels

    def __getitem__(self, idx):
        img = self._data[idx]
        label = self._label[idx]
        return img, label

    def __len__(self):
        return len(self._data)
