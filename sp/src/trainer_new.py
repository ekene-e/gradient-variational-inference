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
from binary_opt import Binary

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.data_loader = Data_Loader(args.data_dir)
        
        # true annotation weight vector and true causal SNPs
        self.true_w, self.true_cs = self.data_loader.global_params()
        self.num_annotations = self.true_w.shape[0]
        # annotation weight vector to be learned as a parameter
        self.w = self.init_weight_vec()
        
        print(self.w)
        
        # optimizer for annotation weight vector w
        if self.args.weight_opt == 'adam':
            self.weight_opt = torch.optim.Adam([self.w], 
                                                maximize=True,
                                                lr=args.lr,
                                                weight_decay=args.weight_decay)
        elif self.args.weight_opt == 'binary':
            self.weight_opt = Binary([self.w])
        
        # list of (SparsePro model, SparsePro optimizer) tuples
        self.model_list = self.init_models()
        
    def init_weight_vec(self):
        self.rng = torch.Generator()
        self.rng.manual_seed(self.args.seed)
        
        mean = torch.zeros(self.num_annotations)
        std = torch.eye(self.num_annotations)
        w = torch.normal(mean=mean, std=std, generator=self.rng).diag()
        return nn.Parameter(w)

    def init_models(self):
        model_list = []
        self.total_num_SNPs = 0
        
        for locus in range(self.args.num_loci):
            X, y, A, n, p = self.data_loader.locus_data(locus) # load locus data
            model = SparsePro(X, y, p, n, self.w, A, self.args.max_num_effects)
            
            if self.args.variational_opt == 'adam':
                opt = torch.optim.Adam(
                    model.parameters(),
                    maximize=True,
                    lr=self.args.lr,
                    weight_decay=self.args.weight_decay)
            elif self.args.variational_opt == 'cavi':
                opt = CAVI(model.parameters(), model)
                
            model_list.append((model, opt))
            self.total_num_SNPs += p
        return model_list
    
    def train(self):
        prev = torch.tensor([0])
        for epoch in range(self.args.num_epochs):
            if self.args.verbose and epoch == 0: print('ELBO\n', '-'*80)
            
            for locus in range(self.args.num_loci):
                print(self.w)
                # load model and optimizer for cur locus
                self.model, self.variational_opt = self.model_list[locus]
                self.model.train()
                self.model.update_pi(self.w)
                
                for iter in range(self.args.num_steps):
                    # update SparsePro model parameters
                    self.variational_opt.zero_grad()
                    loss = self.model()
                    loss.backward()
                    self.variational_opt.step()
                    
                    # update weight vector
                    self.weight_opt.zero_grad()
                    loss = self.model()
                    loss.backward()
                    if self.args.weight_opt == 'binary':
                        self.weight_opt.update_model(self.model)
                    self.weight_opt.step()

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
            if locus in self.true_cs:
                true_idx = torch.tensor(self.true_cs[locus])
                true[prev + true_idx] = 1
            prev += self.model.p

        print(self.w, self.true_w)
        self.plot_auprc(true.detach().numpy(), pred.detach().numpy())
        
    def plot_elbo(self):
        pass

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
            f'___opt-{self.args.variational_opt}'
            f'___opt-{self.args.weight_opt}'
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