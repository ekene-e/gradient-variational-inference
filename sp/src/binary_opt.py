import torch
from torch.optim import Optimizer

class Binary(Optimizer):
    def __init__(self, params):
        defaults = {} # empty dict
        super(Binary, self).__init__(params, defaults)
        
    def update_model(self, model):
        # annotation matrix A and variational parameter gamma 
        self.A = model.annotations # [num_snp x num_annotations]
        self.gamma = model.gamma() # [num_snp x 1]
                
        # hyperparameters
        self.k = model.k # num causal effects
        self.num_snp = self.A.shape[0] # num of SNPs (in this locus)
        self.num_annotations = self.A.shape[1] # num annotations
        self.softmax = model.softmax # softmax over dimension 0
    
    @torch.no_grad()
    def step(self, closure=None):
        w = self.param_groups[0]['params'][0]
        t1 = self.A @ w # compute once outside for loop
        
        # iterate over all annotations
        for i in range(self.num_annotations):
            t2 = self.A[:,i] * w[i]
            
            idx0 = (self.A[:,i] == 0).int()
            idx1 = (self.A[:,i] == 1).int()
            
            k0 = torch.sum(idx0 * self.softmax(t1 - t2))
            k1 = torch.sum(idx1 * self.softmax(t1 - t2))
            
            r0 = torch.sum(idx0 * self.gamma[:, i])
            r1 = torch.sum(idx1 * self.gamma[:, i])
            
            w[i] = torch.log((r1/r0) / (k1/k0))