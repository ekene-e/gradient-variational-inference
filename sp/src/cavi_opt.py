import torch
from torch.optim import Optimizer

class CAVI(Optimizer):
    def __init__(self, params, model):

        self.k = model.k
        self.ytX = model.ytX
        self.XtX = model.XtX
        self.beta_post_tau = model.beta_post_tau
        self.y_tau = model.y_tau
        self.pi = model.pi
        self.softmax = model.softmax
        
        defaults = {} # empty dict
        super(CAVI, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        u = self.param_groups[0]['params'][0]
        gamma = self.softmax(u)
        beta_mu = self.param_groups[0]['params'][1]
                
        for k in range(self.k):
            idxall = [x for x in range(self.k)]
            idxall.remove(k)
            beta_all_k = (gamma[:,idxall] * beta_mu[:,idxall]).sum(axis=1)
            beta_mu[:,k] = (self.ytX-torch.matmul(beta_all_k, self.XtX))/self.beta_post_tau[:,k] * self.y_tau
            u[:,k] = -0.5*torch.log(self.beta_post_tau[:,k]) + torch.log(self.pi.t()) + 0.5 * beta_mu[:,k]**2 * self.beta_post_tau[:,k]
            gamma[:,k] = self.softmax(u[:,k])
            
        assert(beta_mu.isnan().any() == False)
        assert(u.isnan().any() == False)