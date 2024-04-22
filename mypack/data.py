from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_list):
        super().__init__()

        self.data_list = data_list


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx):
        data, label = self.data_list[idx]

        return data.view(-1), label.view(-1)
