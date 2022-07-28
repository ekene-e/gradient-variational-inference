import os
import argparse
from trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SparsePro- for Simulated Data')

    # Directory Argument
    parser.add_argument('--data-dir', type=str, required=True) 
    parser.add_argument('--save-dir', type=str, required=True)

    # Model Argument
    parser.add_argument('--max-num-effects', type=int, default=9)

    # Training Argument
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max-iter', type=int, default=1000)
    parser.add_argument('--eps', type=float, default=1e-5,
        help='threshold for loss improvement')
    parser.add_argument('--weight-decay', type=float, default=5e-3)

    # System Argument
    parser.add_argument('--seed', type=int, default=10)

    # Unsure If Necessary
    parser.add_argument('--prefix', type=str, required=True, 
        help='prefix for result files')
    parser.add_argument("--tmp", action="store_true",
        help='options for saving intermediate file')
    parser.add_argument("--ukb", action="store_true",
        help='use precomputed UK Biobank ld files from PolyFun')


    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    trainer = Trainer(args)
    trainer.train()