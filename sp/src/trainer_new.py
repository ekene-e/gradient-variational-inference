import os
from tqdm import trange

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.metrics import PrecisionRecallDisplay 
import torch.nn as nn


from model import SparsePro
from data import Data_Loader
from cavi_opt import CAVI

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.data_loader = Data_Loader(args.data_dir)
        
        self.w, self.cs_idx = self.data_loader.global_params()
        print(self.cs_idx)
        
        self.weight_vec_optimizer = torch.optim.Adam([self.w], 
                                            maximize=True,
                                            lr=args.lr,
                                            weight_decay=args.weight_decay)
        
        self.model_list = self.init_models()

    def init_models(self):
        model_list = []
        self.total_num_SNPs = 0
        
        for locus in range(self.args.num_loci):
            X, y, A, n, p = self.data_loader.locus_data(locus) # load locus data
            model = SparsePro(X, y, p, n, self.w, A, self.args.max_num_effects)
            
            if self.args.opt == 'adam':
                opt = torch.optim.Adam(
                    model.parameters(),
                    maximize=True,
                    lr=self.args.lr,
                    weight_decay=self.args.weight_decay)
            elif self.args.opt == 'cavi':
                opt = CAVI(model.parameters(), model)
                
            model_list.append((model, opt))
            self.total_num_SNPs += p
        return model_list
    
    def train(self):

        prev = torch.tensor([0])
        for epoch in range(self.args.num_epochs):
            if self.args.verbose and epoch == 0: print('ELBO\n', '-'*80)
            
            for locus in range(self.args.num_loci):
                # load model and optimizer for cur locus
                self.model, self.sp_optimizer = self.model_list[locus]
                self.model.train()
                self.model.update_pi(self.w)
                
                for iter in range(self.args.num_steps):
                
                    # update SparsePro model parameterrs
                    self.sp_optimizer.zero_grad()
                    loss = self.model()
                    loss.backward()
                    self.sp_optimizer.step()
                    
                    # update weight vector
                    loss = self.model()
                    loss.backward()
                    self.weight_vec_optimizer.step()

                    # print loss
                    if self.args.verbose and epoch % 3 == 0 and iter == self.args.num_steps-1: 
                        print(f'Locus {locus}: ', loss.item())
                    
            # check convergence
            if np.abs(loss.item() - prev.item()) < self.args.eps: break
            prev = loss

    def eval(self):
        pred = torch.zeros((self.total_num_SNPs))
        true = torch.zeros((self.total_num_SNPs))
        
        prev = 0
        for locus in range(self.args.num_loci):
            # load model and set to evaluation mode
            self.model, _ = self.model_list[locus]
            self.model.eval()

            # extract gamma, the prior SNP causality vector 
            gamma = self.model.gamma()
            
            # compute multivariate-or function using log-sum-exp trick
            multivariate_or = 1 - torch.exp(torch.sum(torch.log(1 - gamma), dim=1))
            
            # update pred and true for this locus
            pred[prev:prev + self.model.p] = multivariate_or
            if locus in self.cs_idx:
                true_idx = torch.tensor(self.cs_idx[locus])
                true[prev + true_idx] = 1
            prev += self.model.p

        self.plot_auprc(true.detach().numpy(), pred.detach().numpy())

    def plot_auprc(self, true, pred):
        '''Plot Area Under Precision Recall Curve (AUPRC)

        AUPRC is a popular binary classification metric, outputing a scalar
        taking into account precision and recall. 

        Parameters
        ----------
        true : tensor [num_true_causal_SNPs x 1]
            true causal SNPs, obtained from simulated data generation
        pred : tensor [num_predicted_causal_SNPs x 1]
            predicted causal SNPs, obtained where gamma > causality_threshold
        '''

        disp = PrecisionRecallDisplay.from_predictions(true, pred)
        
        plot_dir = 'res/plots' # relative path to directory of saved plots
        filename = ('AUPRC'
            f'___opt-{self.args.opt}'
            f'_lr-{self.args.lr}'
            f'_max-iter-{self.args.num_epochs}'
            f'_eps-{self.args.eps}'
            '.png'
        )

        # save AUPRC plot
        plt.savefig(os.path.join(plot_dir, filename))

        # show AUPRC plot
        if self.args.verbose:
            plt.show()