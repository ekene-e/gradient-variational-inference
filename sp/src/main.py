import os
import argparse
from trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser('SparsePro- for Simulated Data')

    # Directory Argument
    parser.add_argument('--data-dir', type=str, default='data') 
    parser.add_argument('--save-dir', type=str, default='res')

    # Model Argument
    parser.add_argument('--max-num-effects', type=int, default=20)
    parser.add_argument('--annotations', action='store_true')

    # Training Argument
    parser.add_argument('--variational-opt', choices=['adam', 'cavi'], required=True)
    parser.add_argument('--weight-opt', choices=['adam', 'binary'], required=True)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-epochs', type=int, default=3)
    parser.add_argument('--num-steps', type=int, default=30)
    parser.add_argument('--eps', type=float, default=1e-7,
        help='threshold for loss improvement')
    parser.add_argument('--weight-decay', type=float, default=5e-3)

    # System Argument
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--verbose', action='store_true')
    
    # Dataset Argument
    parser.add_argument('--num-loci', type=int,default=10)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    trainer = Trainer(args)
    trainer.train()
    trainer.eval()