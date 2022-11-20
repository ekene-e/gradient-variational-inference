import os
import numpy as np
import torch

class Data_Loader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
        # self.p = 1229 # num SNPs
        # self.n = 322 # num individuals

        # # load data
        # X = np.load(os.path.join(data_dir, 'genotype.npy'))
        # y = np.load(os.path.join(data_dir, 'simulated_phenotype.npy'))
        # snp_classification = np.load(os.path.join(data_dir, 'snp_classification.npy'))

        # # convert to tensors
        # self.X = torch.tensor(X, dtype=torch.float32)
        # self.y = torch.tensor(y, dtype=torch.float32)
        # self.snp_classification = torch.tensor(snp_classification, dtype=torch.float32)
        
    def global_params(self):
        data = np.load(os.path.join(self.data_dir, 'global_params.npz'))
        w = torch.tensor(data['weight'], dtype=torch.float32)
        y = torch.tensor(data['phenotype'], dtype=torch.float32)
        cs_idx = torch.tensor(data['cs_idx'], dtype=torch.float32)
        return w, y, cs_idx
        
    def locus_data(self, locus_num):
        data = np.load(os.path.join(self.data_dir, f'loci_{locus_num}.npz'))
        
        X = torch.tensor(data['genotype'], dtype=torch.float32)
        A = torch.tensor(data['annotation'], dtype=torch.float32)
        n = torch.tensor(X.shape[0], dtype=torch.int)
        p = torch.tensor(X.shape[1], dtype=torch.int)
        
        return X, A, n, p
    