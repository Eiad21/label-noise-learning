import argparse
import logging
import numpy as np
import torch
import os
import sys

from data_classes.data import Dataset
from train import set_optimizer, train_one_epoch, test_one_epoch
from model import Net

# --------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Training DNNs on Imag Datasets')
parser.add_argument("--device", type=str, default="cuda", metavar="D", help="GPU ID")
parser.add_argument("--dataset", type=str, default="cifar100", help="dataset name")
parser.add_argument("--data-dir", type=str, default="./data", help="data directory(./data)")

parser.add_argument("--noise-type", type=str, default='instance', help="type of label noise")
parser.add_argument("--noise-rate", type=float, default=0.4, help="noise rate for label noise")
parser.add_argument("--label-corr", type=bool, default=False, help="use label correction")

parser.add_argument("--batch-size", type=int, default=128, metavar="BS", help="batch size (64)")
parser.add_argument("--num-classes", type=int, default=8, metavar="S", help="num of classes for this dataset")

parser.add_argument("--model", type=str, default='resnet18', metavar="M", help="training model")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="SGD momentum")
parser.add_argument("--lr", type=float, default=0.02, metavar="LR", help="learning rate")
parser.add_argument("--epochs", type=int, default=300, metavar="N", help="number of training epochs")
parser.add_argument("--gamma", type=float, default=0.1, metavar="N", help="gamma for lr scheduler")
parser.add_argument("--weight-decay", default=5e-4, type=float, metavar="WD", help="optimizer weight decay")

args = parser.parse_args()
device = torch.device(args.device)

def save_stats(stats_array,
               path_base,
               path_dict):

    os.makedirs(path_base, exist_ok=True)
    PATH = path_base + 'stats_'
    for key in path_dict:
        PATH += key + '_' + str(path_dict[key]) + '_'
    torch.save(stats_array, PATH)
    print("Saved stats in {} ".format(PATH))


def main():
    print(args)
    dataset = Dataset(args.dataset,
                      args.data_dir,
                      args.noise_type,
                      args.noise_rate,
                      random_seed=1,
                      device=device)

    train_dataloader = torch.utils.data.DataLoader(dataset=dataset.train_set,
                                                   batch_size=args.batch_size,
                                                   shuffle=True)

    test_dataloader = torch.utils.data.DataLoader(dataset=dataset.test_set,
                                                  batch_size=args.batch_size,
                                                  shuffle=False)
    
    net = Net(args.model,
              dataset.num_classes)
    model = net.model.to(device)
    

    optimizer, scheduler = set_optimizer(dataset_name=args.dataset,
                                         model=model,
                                         learning_rate=args.lr,
                                         all_epochs=args.epochs,
                                         gamma=args.gamma,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)


    for epoch in range(args.epochs):

        print("Epoch : {}".format(epoch))

        train_one_epoch(net.model,
                        train_dataloader,
                        optimizer,
                        device,
                        epoch)
        
        scheduler.step()

        test_one_epoch(net.model,
                       test_dataloader,
                       epoch,
                       device,
                       args.num_classes)
        
        sys.stdout.flush()
  

if __name__ == '__main__':
    main()
