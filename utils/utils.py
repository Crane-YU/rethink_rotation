import os
import torch
import random
import torch.nn.functional as F
import numpy as np


def init_folder(args):
    os.makedirs(args.ckpt_path, exist_ok=True)
    exp_path = os.path.join(args.ckpt_path, args.exp_name)
    os.makedirs(exp_path, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_path, args.exp_name, 'ckpts')
    os.makedirs(ckpt_path, exist_ok=True)

    model_name = 'model_cls.py' if args.mode == 'train' else 'model_seg.py'
    dataset_name = 'dataset_modelnet.py' if args.dataset == 'modelnet' else 'dataset_scanobjectnn.py'
    os.system(f'cp main_cls.py {exp_path}/main.py')
    os.system(f'cp {dataset_name} {exp_path}/data.py')
    os.system(f'cp {model_name} {exp_path}/model.py')
    os.system(f'cp -r utils/*.py {exp_path}/')

    return exp_path, ckpt_path


def seed_everything(args):
    seed = args.seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes, device=y.device)[y.cpu().data.numpy()]
    return new_y


def cal_loss(pred, gold, smoothing=True):
    """
    Calculate cross entropy loss, apply label smoothing if needed.
    :param pred:
    :param gold:
    :param smoothing:
    :return:
    """
    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()
