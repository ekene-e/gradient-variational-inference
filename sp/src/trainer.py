import numpy as np
import torch
from sklearn.metrics import PrecisionRecallDisplay 
import matplotlib.pyplot as plt

from model import SparsePro
from data import Data
from cavi_opt import CAVI

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.data = Data(args.data_dir)

        self.model = SparsePro(
            self.data.X,
            self.data.y,
            self.data.p,
            self.data.n,
            args.max_num_effects)

        if args.opt == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                maximize=True,
                lr=args.lr,
                weight_decay=args.weight_decay)
        elif args.opt == 'cavi':
            self.optimizer = CAVI(self.model.parameters(), self.model)

    def train(self):
        self.model.train()

        prev = torch.tensor([0])
        for epoch in range(self.args.max_iter):
            self.optimizer.zero_grad()
            loss = self.model()
            loss.backward()
            self.optimizer.step()

            # print loss
            temp = torch.argwhere(torch.any(self.model.gamma() > self.args.causality_threshold, axis=1))
            if self.args.verbose and epoch == 0: print('\tELBO\t\t\t\tPredicted Casual SNPs\n', '-'*80)
            if self.args.verbose and epoch % 10 == 0: print(f'{loss.item()}\t\t{temp.T}')
            
            # check convergence
            if np.abs(loss.item() - prev.item()) < self.args.eps: break
            prev = loss

    def eval(self):
        self.model.eval()

        # extract relevant variables
        gamma = self.model.gamma()
        causality_thresh = self.args.causality_threshold

        # predictions of casual SNPs 
    
        ''' # pred idx from top k gamma values --> pretty sure this is wrong
        val, flatten_idx = torch.topk(gamma.flatten(), k=10, sorted=True) # top k 
        pred_idx = flatten_idx % self.model.p
        '''

        # pred idx from gamma values > causality_threshold
        pred_idx = torch.argwhere(torch.any(gamma > causality_thresh, axis=1)).T
        pred = torch.zeros(self.data.p)
        pred[pred_idx] = 1

        # true casual SNPs
        true = self.data.snp_classification
        true_idx = torch.argwhere(true).T

        if self.args.verbose: print('\n\nPredicted Casual SNPs:\t', np.sort(pred_idx.data),
                                    '\nTrue Casual SNPs:\t', np.sort(true_idx))
        self.eval_helper(true, pred)

    def eval_helper(self, true, pred):
        disp = PrecisionRecallDisplay.from_predictions(true, pred)
        plt.show()