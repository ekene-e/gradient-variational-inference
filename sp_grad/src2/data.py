import os
import numpy as np

class Data:
    def __init__(self, data_dir):
        self.p = 1229 # num SNPs
        self.n = 322 # num individuals

        self.X = np.load(os.path.join(data_dir, 'genotype.npy'))
        self.y = np.load(os.path.join(data_dir, 'simulated_phenotype.npy'))
        self.casual_snps = np.load(os.path.join(data_dir, 
                                                    'casual_snp_idx.npy'))