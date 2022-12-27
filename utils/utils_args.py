import argparse
import torch


def get_args_base():
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--ckpt_path', type=str, default='ckpt',
                        help='checkpoint path')
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='name of the experiment')
    parser.add_argument('--dataset', type=str, choices=['modelnet', 'scanobject'],
                        help='name of the dataset')
    parser.add_argument('--cuda', action='store_true',
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=0, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--k', type=int, default=32,
                        help='num of neighboring points')
    parser.add_argument('--sigma', type=int, default=15,
                        help='t')
    parser.add_argument('--emb_dims', type=int, default=1024,
                        help='embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--pca_mode', type=str, default='nearest_v1',
                        help='name of the pca mode')
    return parser


def get_args_train():
    parser = get_args_base()
    parser.add_argument('--train_batch_size', type=int, default=32,
                        help='size of train batch)')
    parser.add_argument('--test_batch_size', type=int, default=16,
                        help='size of test batch)')
    parser.add_argument('--epochs', type=int, default=250,
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action='store_true',
                        help='use SGD')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='scheduler',
                        help='use SGD')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.0001, 0.01 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--wt_decay', type=float, default=5e-4, metavar='wd',
                        help='weight decay')
    parser.add_argument('--use_contrast', action='store_true',
                        help='use contrastive learning')
    parser.add_argument('--train_rot_mode', type=str, default='z', choice=['z', 'so3'],
                        help='name of different rotations')
    parser.add_argument('--test_rot_mode', type=str, default='z', choice=['z', 'so3'],
                        help='name of different rotations')
    args = parser.parse_args()
    return args, parser


def get_args_test():
    parser = get_args_base()
    parser.add_argument('--test_batch_size', type=int, default=16,
                        help='size of test batch)')
    parser.add_argument('--test_rot_mode', type=str, default='z', choice=['z', 'so3'],
                        help='name of different rotations')
    args = parser.parse_args()
    return args, parser


def process_args(train=True):
    if train:
        args, _ = get_args_train()
        args.mode = 'train'
    else:
        args, _ = get_args_test()
        args.mode = 'test'
    args.cuda = args.cuda and torch.cuda.is_available()
    return args
