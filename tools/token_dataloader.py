# Load data from token npy files
import glob
import numpy as np
import torch


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


class CondTokenLoader(UncondTokenLoader):

    def __iter__(self):
        while self.end < len(self.data_paths):
            batch_paths = self.data_paths[self.start:self.end]
            cbg_tokens = []
            cid_tokens = []
            cmo_tokens = []
            xbg_tokens = []
            xid_tokens = []
            xmo_tokens = []

            for path in batch_paths:
                data = np.load(path, allow_pickle=True).item()
                cbg_tokens.append(data['cbg_tokens'])
                cid_tokens.append(data['cid_tokens'])
                cmo_tokens.append(data['cmo_tokens'])
                xbg_tokens.append(data['bg_tokens'])
                xid_tokens.append(data['id_tokens'])
                xmo_tokens.append(data['mo_tokens'])

            cbg_tokens = torch.from_numpy(np.array(cbg_tokens)).to(self.device)
            cid_tokens = torch.from_numpy(np.array(cid_tokens)).to(self.device)
            cmo_tokens = torch.from_numpy(np.array(cmo_tokens)).to(self.device)
            xbg_tokens = torch.from_numpy(np.array(xbg_tokens)).to(self.device)
            xid_tokens = torch.from_numpy(np.array(xid_tokens)).to(self.device)
            xmo_tokens = torch.from_numpy(np.array(xmo_tokens)).to(self.device)
            self.end += self.batch_size
            self.start += self.batch_size
            c_toks = torch.unsqueeze(cbg_tokens, dim=1), torch.unsqueeze(cid_tokens, dim=1), cmo_tokens
            x_toks = torch.unsqueeze(xbg_tokens, dim=1), torch.unsqueeze(xid_tokens, dim=1), xmo_tokens
            yield c_toks, x_toks


if __name__ == '__main__':
    # data_folder_path = '../data2'
    # paths = glob.glob(data_folder_path + '/*.npy')
    # path = paths[0]
    # data = np.load(path, allow_pickle=True).item()
    # print(data.keys())
    loader = CondTokenLoader('/export2/xu1201/MOSO/merged_Token/UCF101/img256_16frames/train', batch_size=2)
    for batch in loader:
        print(batch)
        break