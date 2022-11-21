import os
import numpy as np
import torch

class Data_Loader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
    def global_params(self):
        data = np.load(os.path.join(self.data_dir, 'global_params.npz'), allow_pickle=True)
        w = torch.tensor(data['weight'], dtype=torch.float32)
        cs_idx = data['cs_idx'].item() # dict of lists
        return w, cs_idx
        
    def locus_data(self, locus_num):
        data = np.load(os.path.join(self.data_dir, f'loci_{locus_num}.npz'))
        
        X = torch.tensor(data['genotype'], dtype=torch.float32)
        y = torch.tensor(data['phenotype'], dtype=torch.float32)
        A = torch.tensor(data['annotation'], dtype=torch.float32)
        n = torch.tensor(X.shape[0], dtype=torch.int)
        p = torch.tensor(X.shape[1], dtype=torch.int)
        
        return X, y, A, n, p
    