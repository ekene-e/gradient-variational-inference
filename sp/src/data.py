import os
import numpy as np
import torch

class Data:
    def __init__(self, data_dir):
        self.p = 1229 # num SNPs
        self.n = 322 # num individuals

        # load data
        X = np.load(os.path.join(data_dir, 'genotype.npy'))
        y = np.load(os.path.join(data_dir, 'simulated_phenotype.npy'))
        snp_classification = np.load(os.path.join(data_dir, 'snp_classification.npy'))

        # convert to tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.snp_classification = torch.tensor(snp_classification, dtype=torch.float32)