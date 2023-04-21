# Load data from token npy files
import glob
import numpy as np
import torch


class TokenLoader:

    def __init__(self, data_folder_path, batch_size=1, shuffle=True):
        paths = glob.glob(data_folder_path + '/*.npy')
        self.data_paths = np.array(paths)
        self.batch_size = batch_size
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

            bg_tokens = torch.from_numpy(np.array(bg_tokens))
            id_tokens = torch.from_numpy(np.array(id_tokens))
            mo_tokens = torch.from_numpy(np.array(mo_tokens))
            self.end += self.batch_size
            self.start += self.batch_size
            yield torch.unsqueeze(bg_tokens, dim=1), torch.unsqueeze(id_tokens, dim=1), mo_tokens



if __name__ == '__main__':
    loader = TokenLoader('../data', batch_size=2)
    for batch in loader:
        bg, id, mo = batch
        print(bg.shape, id.shape, mo.shape)