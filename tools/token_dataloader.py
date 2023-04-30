# Load data from token npy files
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
# import torch.utils.data as Data


class UncondTokenLoader:

    def __init__(self, data_folder_path, batch_size=1, device='cpu', shuffle=True):
        paths = glob.glob(data_folder_path + '/*/*/*.npy')
        self.data_paths = np.array(paths)
        self.batch_size = batch_size
        self.device = device
        if shuffle:
            self.data_paths = np.random.permutation(self.data_paths)

        self.end = self.batch_size
        self.start = 0

    def reset(self, shuffle=False):
        if shuffle:
            self.data_paths = np.random.permutation(self.data_paths)
        self.end = self.batch_size
        self.start = 0

    def __iter__(self):
        while self.end < len(self.data_paths):
            batch_paths = self.data_paths[self.start:self.end]
            bg_tokens = []
            id_tokens = []
            mo_tokens = []
            for path in batch_paths:
                data = np.load(path, allow_pickle=True).item()
                bg_tokens.append(data['bg_tokens'])
                id_tokens.append(data['id_tokens'])
                mo_tokens.append(data['mo_tokens'])

            bg_tokens = torch.from_numpy(np.array(bg_tokens)).to(self.device)
            id_tokens = torch.from_numpy(np.array(id_tokens)).to(self.device)
            mo_tokens = torch.from_numpy(np.array(mo_tokens)).to(self.device)
            self.end += self.batch_size
            self.start += self.batch_size
            yield torch.unsqueeze(bg_tokens, dim=1), torch.unsqueeze(id_tokens, dim=1), mo_tokens


class CondTokenDataset(Dataset):

    def __init__(self, data_folder_path, device='cpu', shuffle=True):
        self.data_folder_path = data_folder_path
        self.device = device
        self.items = os.listdir(data_folder_path)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):

        path = self.data_folder_path + '/*/*/' + self.items[index]
        data = np.load(path, allow_pickle=True).item()
        bg_tokens = torch.from_numpy(np.array(data['bg_tokens'])).to(self.device)
        id_tokens = torch.from_numpy(np.array(data['id_tokens'])).to(self.device)
        mo_tokens = torch.from_numpy(np.array(data['mo_tokens'])).to(self.device)

        cbg_tokens = torch.from_numpy(np.array(data['cbg_tokens'])).to(self.device)
        cid_tokens = torch.from_numpy(np.array(data['cid_tokens'])).to(self.device)
        cmo_tokens = torch.from_numpy(np.array(data['cmo_tokens'])).to(self.device)

        c_toks = torch.unsqueeze(cbg_tokens, dim=0), torch.unsqueeze(cid_tokens, dim=0), cmo_tokens
        x_toks = torch.unsqueeze(bg_tokens, dim=0), torch.unsqueeze(id_tokens, dim=0), mo_tokens

        return c_toks, x_toks


if __name__ == '__main__':
    # data_folder_path = '../data2'
    # paths = glob.glob(data_folder_path + '/*.npy')
    # path = paths[0]
    # data = np.load(path, allow_pickle=True).item()
    # print(data.keys())

    # loader = CondTokenLoader('/export2/xu1201/MOSO/merged_Token/UCF101/img256_16frames/train', batch_size=2)
    # for batch in loader:
    #     print(batch)
    #     break

    tokens_dir = '../data2'
    loader = CondTokenDataset(tokens_dir)

    d = DataLoader(loader, batch_size=3, shuffle=False)
    # print(len(loader))
    for it, (c, x) in enumerate(d):
        cbg_tokens, cid_tokens, cmo_tokens = c
        xbg_tokens, xid_tokens, xmo_tokens = x
        print(it, cbg_tokens.shape, cid_tokens.shape, cmo_tokens.shape, xbg_tokens.shape, xid_tokens.shape, xmo_tokens.shape)

