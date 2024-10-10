import argparse
import numpy as np
import torch
import os
import sys

from data_classes.data import Dataset
from train import set_optimizer, train_one_epoch, test_one_epoch, eval_train
from model import Net
from torch.utils.data import SubsetRandomSampler
from collections import Counter

# --------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Training DNNs on Imag Datasets')
parser.add_argument("--device", type=str, default="cuda", metavar="D", help="GPU ID")
parser.add_argument("--dataset", type=str, default="cifar100", help="dataset name")
parser.add_argument("--data-dir", type=str, default="./data", help="data directory(./data)")


parser.add_argument("--noise-type", type=str, default='instance', help="type of label noise")
parser.add_argument("--noise-rate", type=int, default=40, help="noise rate for label noise")
parser.add_argument("--num-classes", type=int, default=8, metavar="S", help="num of classes for this dataset")

parser.add_argument("--batch-size", type=int, default=128, metavar="BS", help="batch size (64)")
parser.add_argument("--num-batches", type=int, default=10000, metavar="B", help="number of mini batches (1000)")
parser.add_argument("--clustering", type=str, default="none", help="how to cluster clean and noisy")
parser.add_argument("--num-samples", type=int, default=50000, metavar="S", help="num of training samples bs*num_batch")


parser.add_argument("--model", type=str, default='preact-resnet18', metavar="M", help="training model")
parser.add_argument("--pretrain", type=bool, default=False, help="use pretrained models")
parser.add_argument("--optimizer", type=str, default='sgd', metavar="OPT", help="optimizer")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="SGD momentum")
parser.add_argument("--lr", type=float, default=0.02, metavar="LR", help="learning rate")
parser.add_argument("--epochs", type=int, default=300, metavar="N", help="number of training epochs")
parser.add_argument("--gamma", type=float, default=0.1, metavar="N", help="gamma for lr scheduler")
parser.add_argument("--weight-decay", default=5e-4, type=float, metavar="WD", help="optimizer weight decay")

args = parser.parse_args()
device = torch.device(args.device)

alpha_schedule = np.zeros(args.epochs)

def save_stats(stats_array,
               path_base,
               path_dict):

    os.makedirs(path_base, exist_ok=True)
    PATH = path_base + 'stats_'
    for key in path_dict:
        PATH += key + '_' + str(path_dict[key]) + '_'
    torch.save(stats_array, PATH)
    print("Saved stats in {} ".format(PATH))

torch.cuda.manual_seed(2)
np.random.seed(2)
def main():
    print(args)
    dataset = Dataset(args.dataset,
                      args.data_dir,
                      args.noise_type,
                      args.noise_rate,
                      random_seed=1,
                      device=device)
    dataset.split_into_batches(args.batch_size, train_sampler=dataset.train_sampler)
    
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

  
    if args.dataset == 'cifar10':
        T_w = 25
        init_alpha = 0.1
    elif args.dataset == 'cifar100':
        T_w = 30
        init_alpha = 0.2
    elif args.dataset == 'tissue':
        T_w = 1
        init_alpha = 0.1
    elif args.dataset == 'chest':
        T_w = 4
        init_alpha = 0.05

    alpha_schedule[0:T_w] = np.linspace(init_alpha, 0, T_w)
    clean_labels_np = dataset.clean_labels.detach().numpy()

    # clean_indices = np.where(clean_labels_np == dataset.get_targets().detach().numpy())[0]
    clean_indices = None

    for epoch in range(args.epochs):
        all_idx = np.arange(0, args.num_samples)
        train_sampler = SubsetRandomSampler(all_idx)
        dataset.train_dataloader = torch.utils.data.DataLoader(dataset=dataset.train_set,
                                                               batch_size=args.batch_size,
                                                               sampler=train_sampler)
        epoch_alpha = alpha_schedule[epoch]
        print('Current sieving threshold is {}'.format(epoch_alpha))


        (clean_labels_epoch, noisy_labels_epoch, 
         _, _) = eval_train(net.model, epoch, dataset.train_dataloader, 
                                args.num_samples, epoch_alpha, dataset.num_classes, 
                                args.clustering, device, clean_indices, clean_labels_np)

        calculate_filtering_success(dataset, clean_labels_np, clean_labels_epoch, noisy_labels_epoch)

        clean_labels_epoch_np = clean_labels_epoch.numpy()
        clean_labels_epoch = torch.from_numpy(clean_labels_epoch_np)

        np.random.shuffle(clean_labels_epoch)
        selected_idx = clean_labels_epoch
        
        aug_n = int(len(noisy_labels_epoch))

        if aug_n < len(clean_labels_epoch):
            new_selected_idx = torch.cat((selected_idx, selected_idx[0:aug_n]))
        else:
            q = aug_n // len(clean_labels_epoch)
            r = aug_n % len(clean_labels_epoch)
            new_selected_idx = torch.Tensor()
            while q >= 0:
                new_selected_idx = torch.cat((new_selected_idx, clean_labels_epoch))
                q = q - 1
            if r != 0:
                new_selected_idx = torch.cat((new_selected_idx, clean_labels_epoch[0:r]))

        train_sampler = SubsetRandomSampler(new_selected_idx.to(torch.int64))
        train_dataloader = torch.utils.data.DataLoader(dataset=dataset.train_set,
                                                       batch_size=args.batch_size,
                                                       sampler=train_sampler)


        print("Epoch : {}".format(epoch))
        
        train_one_epoch(net.model,
                        train_dataloader,
                        optimizer,
                        device,
                        epoch)
        
        scheduler.step()

        test_one_epoch(net.model,
                       dataset.test_dataloader, 
                       epoch,
                       device,
                       dataset.num_classes)
        
        sys.stdout.flush()

def calculate_filtering_success(dataset, clean_labels_np, clean_labels_epoch, noisy_labels_epoch):
    good_clean = 0
    good_noisy = 0

    for idx in clean_labels_epoch:
        if clean_labels_np[idx] == dataset.get_targets()[idx]:
            good_clean += 1
    for idx in noisy_labels_epoch:
        if clean_labels_np[idx] != dataset.get_targets()[idx]:
            good_noisy += 1

    if len(clean_labels_epoch) > 0:
        print(f"Clean Selection success: {good_clean}/{len(clean_labels_epoch)}  {good_clean/len(clean_labels_epoch)*100}%")
    else:
        print("No clean labels selected")
    if len(noisy_labels_epoch) > 0:
        print(f"Noisy Selection success: {good_noisy}/{len(noisy_labels_epoch)}  {good_noisy/len(noisy_labels_epoch)*100}%")
        label_counts = Counter(clean_labels_np[noisy_labels_epoch])
        print("Counts of filtered labels")
        print(label_counts)
    else:
        print("No noisy labels selected")
        


if __name__ == '__main__':    
    main()
