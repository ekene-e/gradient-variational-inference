import os

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
        
        self.w, self.y, self.cs_idx = self.data_loader.global_params()
        
        self.weight_vec_optimizer = torch.optim.Adam([self.w], 
                                            maximize=True,
                                            lr=args.lr,
                                            weight_decay=args.weight_decay)
        
        self.model_list = self.init_models()

    def init_models(self):
        model_list = []
        for locus in range(self.args.num_loci):
            X, A, n, p = self.data_loader.locus_data(locus) # load locus data
            model = SparsePro(X, self.y, p, n, self.w, A, self.args.max_num_effects)
            
            if self.args.opt == 'adam':
                opt = torch.optim.Adam(
                    model.parameters(),
                    maximize=True,
                    lr=self.args.lr,
                    weight_decay=self.args.weight_decay)
            elif self.args.opt == 'cavi':
                opt = CAVI(model.parameters(), model)
                
            model_list.append((model, opt))
            return model_list
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
                    if self.args.verbose and epoch % 20 == 0: 
                        print(loss.item())
                    return
            # check convergence
            if np.abs(loss.item() - prev.item()) < self.args.eps: break
            prev = loss

        # print loss at convergence
        if self.args.verbose:
            print(f'At iter {epoch}, ELBO converged to {self.model():.4f}')

    def eval(self):
        self.model.eval()

        # extract relevant variables
        gamma = self.model.gamma()
        causality_thresh = self.args.causality_threshold

        # predictions of casual SNPs with gamma values > causality_threshold
        pred_idx = torch.argwhere(torch.any(gamma > causality_thresh, axis=1)).T
        pred = torch.zeros(self.data.p)
        pred[pred_idx] = 1

        # true casual SNPs
        true = self.data.snp_classification
        true_idx = torch.argwhere(true).T

        # print predicted and true causal SNPs
        if self.args.verbose: 
            print(
                '\n\nPredicted Causal SNPs:\t', np.sort(pred_idx.detach().reshape(-1)),
                '\nTrue Causal SNPs:\t', np.sort(true_idx.reshape(-1))
            )

        self.plot_auprc(true, gamma.detach().numpy())
        #self.plot_auprc(true, pred)

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
            f'_causal-thresh-{self.args.causality_threshold}'
            f'_lr-{self.args.lr}'
            f'_max-iter-{self.args.max_iter}'
            f'_eps-{self.args.eps}'
            '.png'
        )

        # save AUPRC plot
        plt.savefig(os.path.join(plot_dir, filename))

        # show AUPRC plot
        if self.args.verbose:
            plt.show()