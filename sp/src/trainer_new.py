import os
from tqdm import trange

import numpy as np
import torch
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay 
import torch.nn as nn

from model import SparsePro
from data import Data_Loader
from cavi_opt import CAVI
from binary_opt import Binary

sns.set_theme()

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.data_loader = Data_Loader(args.data_dir)
        
        # true annotation weight vector and true causal SNPs
        self.true_w, self.true_cs = self.data_loader.global_params()
        print(self.true_cs)
        self.num_annotations = self.true_w.shape[0]
        self.w = None # assume no annotation weight vector by default
        
        # if using functional annotations, learn annotation weight vector
        if self.args.annotations:
            # annotation weight vector to be learned as a parameter
            self.w = self.init_weight_vec()
        
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
            model = SparsePro(X, y, p, n, A, self.w, self.args.max_num_effects)
            
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
        self.elbo = [[] for _ in range(self.args.num_loci)]
        for epoch in range(self.args.num_epochs):
            if self.args.verbose and epoch == 0: print('\t\t\t\tMODEL TRAINING\n', '-'*80)
            print('-'*36, f'EPOCH {epoch}', '-'*36)
            if epoch % 5 == 0: print(self.w, '\n')
            
            for locus in range(self.args.num_loci):
                # load model and optimizer for cur locus
                self.model, self.variational_opt = self.model_list[locus]
                self.model.train()
                if self.args.annotations: self.model.update_pi(self.w)
                
                for iter in range(self.args.num_steps):
                    # update SparsePro model parameters
                    self.variational_opt.zero_grad()
                    loss = self.model()
                    loss.backward(retain_graph=True)
                    self.variational_opt.step()
                    
                    # if learning annotation weight vector
                    if self.args.annotations:
                        # update weight vector
                        self.weight_opt.zero_grad()
                        loss = self.model()
                        loss.backward(retain_graph=True)
                        if self.args.weight_opt == 'binary':
                            self.weight_opt.update_model(self.model)
                        self.weight_opt.step()

                    # print loss
                    self.elbo[locus].append(loss.item())
                    if self.args.verbose and epoch % 5 == 0 and iter == self.args.num_steps-1: 
                        print(f'Locus {locus}: ', self.elbo[locus][-1])
                    
            # check convergence
            if np.abs(self.elbo[locus][-1] - self.elbo[locus][-2]) < self.args.eps: break

    def eval(self):
        print('\n', '-'*35, f'EVALUATION', '-'*35)
        print(self.true_cs)
        self.plot_elbo()
        
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
            val, idx = torch.topk(multivariate_or, 5)
            print(f'Locus {locus}:\t', idx.detach().numpy(), 
                  '  \t\t(SNP Index)', '\n\t\t', 
                  np.around(val.detach().numpy(), decimals=3),
                  '\t(Probability of SNP Causality)')
            
            # update pred and true for this locus
            pred[prev:prev + self.model.p] = multivariate_or
            if locus in self.true_cs:
                true_idx = torch.tensor(self.true_cs[locus])
                true[prev + true_idx] = 1
            prev += self.model.p

        self.plot_auprc(true.detach().numpy(), pred.detach().numpy())
        self.plot_hist1(pred.detach().numpy())
        self.plot_hist2(pred.detach().numpy())
        self.plot_hist3(pred.detach().numpy())        
        
    def plot_elbo(self):
        # plotting
        for locus in range(self.args.num_loci):
            plt.plot(self.elbo[locus], label=f'L{locus}')
        plt.legend()
        plt.xlabel('Training Iteration')
        plt.ylabel('ELBO')
        
        plot_dir = 'res/elbo' # relative path to directory of ELBO plots
        filename = ('ELBO'
            f'__annotations-{self.args.annotations}'
            f'_variational-opt-{self.args.variational_opt}'
            f'_weight-opt-{self.args.weight_opt}'
            f'_lr-{self.args.lr}'
            f'_num-epochs-{self.args.num_epochs}'
            f'_num-steps-{self.args.num_steps}'
            f'_eps-{self.args.eps}'
            f'_seed-{self.args.seed}'
            '.png'
        )
        
        plt.savefig(os.path.join(plot_dir, filename))  # save ELBO plot
        if self.args.verbose: plt.show() # show elbo plot

    def plot_hist1(self, pred):
        idx = 0
        pred_list, labels_list = [], []
        for locus in range(self.args.num_loci):
            self.model, _ = self.model_list[locus]
            cur_pred = pred[idx : idx + self.model.p + 1]
            
            pred_list.append(cur_pred)
            labels_list.append(f'L{locus}')
            idx += self.model.p
            
        plt.hist(pred_list, bins=100, histtype='barstacked', range=(0,1), label=labels_list, lw=0)
        plt.xlabel('Probability SNP Causes (at Least One) Effect')
        plt.ylabel('Frequency')
        plt.legend()
        
        plot_dir = 'res/histogram' # relative path to directory of hist plots
        filename = ('hist1'
            f'__annotations-{self.args.annotations}'
            f'_variational-opt-{self.args.variational_opt}'
            f'_weight-opt-{self.args.weight_opt}'
            f'_lr-{self.args.lr}'
            f'_num-epochs-{self.args.num_epochs}'
            f'_num-steps-{self.args.num_steps}'
            f'_eps-{self.args.eps}'
            f'_seed-{self.args.seed}'
            '.png'
        )
        
        plt.savefig(os.path.join(plot_dir, filename))  # save ELBO plot
        if self.args.verbose: plt.show() # show elbo plot
        
    def plot_hist2(self, pred):
        idx = 0
        pred_list, labels_list = [], []
        for locus in range(self.args.num_loci):
            self.model, _ = self.model_list[locus]
            cur_pred = pred[idx : idx + self.model.p + 1]
            
            pred_list.append(cur_pred)
            labels_list.append(f'L{locus}')
            idx += self.model.p
            
        plt.hist(pred_list, bins=100, histtype='barstacked', range=(0,0.15), label=labels_list, lw=0)
        plt.xlabel('Probability SNP Causes (at Least One) Effect')
        plt.ylabel('Frequency')
        plt.legend()
        
        plot_dir = 'res/histogram' # relative path to directory of hist plots
        filename = ('hist2'
            f'__annotations-{self.args.annotations}'
            f'_variational-opt-{self.args.variational_opt}'
            f'_weight-opt-{self.args.weight_opt}'
            f'_lr-{self.args.lr}'
            f'_num-epochs-{self.args.num_epochs}'
            f'_num-steps-{self.args.num_steps}'
            f'_eps-{self.args.eps}'
            f'_seed-{self.args.seed}'
            '.png'
        )
        
        plt.savefig(os.path.join(plot_dir, filename))  # save ELBO plot
        if self.args.verbose: plt.show() # show elbo plot
        
    def plot_hist3(self, pred):
        idx = 0
        pred_list, labels_list = [], []
        for locus in range(self.args.num_loci):
            self.model, _ = self.model_list[locus]
            cur_pred = pred[idx : idx + self.model.p + 1]
            
            pred_list.append(cur_pred)
            labels_list.append(f'L{locus}')
            idx += self.model.p
            
        plt.hist(pred_list, bins=100, histtype='barstacked', range=(0.9,1), label=labels_list, lw=0)
        plt.xlabel('Probability SNP Causes (at Least One) Effect')
        plt.ylabel('Frequency')
        plt.legend()
        
        plot_dir = 'res/histogram' # relative path to directory of hist plots
        filename = ('hist3'
            f'__annotations-{self.args.annotations}'
            f'_variational-opt-{self.args.variational_opt}'
            f'_weight-opt-{self.args.weight_opt}'
            f'_lr-{self.args.lr}'
            f'_num-epochs-{self.args.num_epochs}'
            f'_num-steps-{self.args.num_steps}'
            f'_eps-{self.args.eps}'
            f'_seed-{self.args.seed}'
            '.png'
        )
        
        plt.savefig(os.path.join(plot_dir, filename))  # save ELBO plot
        if self.args.verbose: plt.show() # show elbo plot
        
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
        
        plot_dir = 'res/auprc' # relative path to directory of AUPRC plots
        filename = ('AUPRC'
            f'__annotations-{self.args.annotations}'
            f'_variational-opt-{self.args.variational_opt}'
            f'_weight-opt-{self.args.weight_opt}'
            f'_lr-{self.args.lr}'
            f'_num-epochs-{self.args.num_epochs}'
            f'_num-steps-{self.args.num_steps}'
            f'_eps-{self.args.eps}'
            f'_seed-{self.args.seed}'
            '.png'
        )

        plt.savefig(os.path.join(plot_dir, filename))  # save AUPRC plot
        if self.args.verbose: plt.show()  # show AUPRC plot