import numpy as np
from sympy import maximum
import torch

from model import SparsePro
from data import Data


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

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            maximize=True,
            lr=args.lr,
            weight_decay = args.weight_decay)

    def train(self):
        self.model.train()

        prev = torch.tensor([0])
        for epoch in range(self.args.max_iter):
            self.optimizer.zero_grad()
            loss = self.model()
            loss.backward()
            self.optimizer.step()

            temp = torch.argwhere(torch.any(self.model.gamma() > 0.1, axis=1))
            if self.args.verbose and epoch % 10 == 0: print(f'{loss.item()}\t\t{temp.T}')
            if np.abs(loss.item() - prev.item()) < self.args.eps: break
            prev = loss

    def eval(self):
        self.model.eval()
 
        gamma = self.model.gamma()

        casuality_threshold = 0.1
        pred_idx = torch.argwhere(torch.any(gamma > casuality_threshold, axis=1))
        pred = torch.zeros(self.data.p)
        pred[pred_idx] = 1

        true = self.data.snp_classification
        true_idx = torch.argwhere(true)

        #print(np.count_nonzero(pred))
        #print(np.count_nonzero(true))
        print(np.sort(pred_idx.T))
        print(np.sort(true_idx.T))