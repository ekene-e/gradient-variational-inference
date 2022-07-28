import numpy as np
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
            lr=args.lr,
            weight_decay = args.weight_decay)

    def train(self):
        self.model.train()

        prev = torch.tensor([0])
        for _ in range(self.args.max_iter):
            self.optimizer.zero_grad()
            loss = self.model()
            loss.backward()
            self.optimizer.step()

            if np.abs(loss.item() - prev.item()) < self.args.eps: break
            prev = loss

    def eval(self):
        self.model.eval()

        gamma = self.model.gamma()
        pred_idx = torch.argwhere(gamma > 0.1) # [? x 2] tensor
        pred = pred_idx[:,1]

        true = self.data.casual_snps