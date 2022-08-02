import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay 
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
                weight_decay = args.weight_decay)
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
            temp = torch.argwhere(torch.any(self.model.gamma() > self.args.casuality_threshold, axis=1))
            if self.args.verbose and epoch % 10 == 0: print(f'{loss.item()}\t\t{temp.T}')
            
            # check convergence
            if np.abs(loss.item() - prev.item()) < self.args.eps: break
            prev = loss

    def eval(self):
        self.model.eval()
 
        gamma = self.model.gamma()

        #pred_idx = torch.argwhere(torch.any(gamma > self.args.casuality_threshold, axis=1))
        val, flatten_idx = torch.topk(gamma.flatten(), k=10, sorted=True) # top k 
        pred_idx = flatten_idx % self.model.p
        pred = torch.zeros(self.data.p)
        pred[pred_idx] = 1
        #pred = pred.reshape(gamma.shape)
        #pred_idx = torch.argwhere(pred)[:,0].reshape(1,-1)

        true = self.data.snp_classification
        true_idx = torch.argwhere(true).T

        if self.args.verbose: print(np.sort(pred_idx.data), '\n', np.sort(true_idx))
        self.eval_helper(true, pred)

    def eval_helper(self, true, pred):
        #precision, recall, thresh = precision_recall_curve(true, pred)
        disp = PrecisionRecallDisplay.from_predictions(true, pred)
        disp.plot()
        plt.show()
