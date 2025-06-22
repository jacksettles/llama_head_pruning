from torch.utils.data import Dataset
import torch

class LlamaDataset(Dataset):
    def __init__(self, data_path):
        self.input_ids = torch.load(data_path)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return torch.tensor(self.input_ids[index], dtype=torch.long)