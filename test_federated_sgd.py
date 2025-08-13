"""
==========================
@author:Zhu Zehan
@time:2020/11/25:18:57
@email:12032045@zju.edu.cn
==========================
"""
import os
import argparse
import numpy as np
import torch.nn as nn

import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import load as cifar

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:55004', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--type', default=1, type=int, metavar='N',
                    help='which algrithom to test')
parser.add_argument('--epoch', default=95, type=int, metavar='N',
                    help='which epoch to test')


class CIFARLoader():
    def __init__(self):
        self.train_data = cifar.x_train
        self.train_label = cifar.y_train
        self.test_data = cifar.x_test
        self.test_label = cifar.y_test

        self.train_data = self.train_data / 255.0
        self.train_label = self.train_label
        self.test_data = self.test_data / 255.0
        self.test_label = self.test_label

        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]


def main():
    args = parser.parse_args()  # 获得命令行参数，若命令行无输入，则使用默认值（default）
    if os.path.exists('./ACC_federated_sgd') is False:
        os.mkdir('./ACC_federated_sgd')
    torch.backends.cudnn.deterministic = True
    mp.spawn(main_worker, nprocs=8, args=(args,))  # 开启4个进程， 每个进程执行main_worker（）函数，且将参数传给该函数


def main_worker(gpu, args):
    """
    进程初始化
    """
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=8, rank=gpu)
    data_loader = CIFARLoader()
    acc = []
    model = nn.Linear(784, 10)
    model.cuda(gpu)
    put_path = './ACC_federated_sgd'
    get_path = './Weights_federated_sgd'

    for epoch in range(80):
        PATH = get_path + '/GPU{}/LeNet_{}.pth'.format(gpu, epoch)
        model.load_state_dict(torch.load(PATH))
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for step in range(100):
                images = data_loader.test_data[step * 100:(step + 1) * 100]
                target = data_loader.test_label[step * 100:(step + 1) * 100]
                torch_images = torch.from_numpy(images).float().cuda(gpu, non_blocking=True)
                torch_target = torch.from_numpy(target).long().cuda(gpu, non_blocking=True)
                # torch_images = torch_images.view(-1, 32 * 32 * 3)
                output = model(torch_images)

                _, predicted = torch.max(output.data, 1)
                total += torch_target.size(0)
                correct += (predicted == torch_target).sum().item()
                print('Accuracy:', 1.0 * correct / total)
        acc.append(1.0 * correct / total)
        print('this is {} epoch'.format(epoch))

    np.save(put_path+'/ACC_federated_sgd_{}.npy'.format(gpu), np.array(acc))


if __name__ == '__main__':
    main()


