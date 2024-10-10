import torch
from tqdm import tqdm
import numpy as np
from torch import optim, nn
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import sys

import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc
from sklearn.cluster import KMeans, MeanShift, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
import seaborn as sns
from collections import Counter


from torchmetrics.classification import Accuracy, F1Score, AUROC, ConfusionMatrix, MulticlassAUROC

# Set optimizer according to the training strategy and
# the type of the optimizer (return optimizer)
def set_optimizer(dataset_name,
                  model,
                  learning_rate,
                  all_epochs,
                  gamma,
                  momentum=0.9,
                  weight_decay=1e-3):
    model_params = model.parameters()

    print(f"Using SGD. LR: {learning_rate}, WD: {weight_decay}, M: {momentum}")
    optimizer = optim.SGD(model_params, learning_rate, weight_decay=weight_decay, momentum=momentum)

    if dataset_name == 'cifar10' or dataset_name == 'cifar100' or dataset_name == 'tissue':
        min_lr = 0.0005
        scheduler = CosineAnnealingLR(optimizer,T_max=all_epochs, eta_min=min_lr)
        print(f'Using CosineAnnealingLR. T_max {all_epochs}, ETA: {min_lr}')

    elif dataset_name == 'clothing1m':
        scheduler = MultiStepLR(optimizer, milestones=40, gamma=gamma, verbose=True)
        print('Set learning rate scheduler to MultiStepLR')
    
    elif dataset_name == 'chest':
        lr_milestones = [7, 14, 21, 28, 35]
        scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=gamma)
        print('Set learning rate scheduler to MultiStepLR')
   
    return optimizer, scheduler


# ---------------------------------------------------------------------------------------------------------------
# Training on batches for only one epoch
# Return(training loss and accuracy)
def train_one_epoch(model,
                    data_loader,
                    optimizer,
                    device,
                    epoch):
    losses = []
    all_target_labels = np.array([], dtype=np.int64)
    all_pred_labels = np.array([], dtype=np.int64)

    print(f'Epoch # {epoch}')

    model.train()

    print("Start train")
    sys.stdout.flush()
    for _, (image, label, _) in enumerate(data_loader):
        all_target_labels = np.append(all_target_labels, label.detach().data)
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(image)

        all_pred_labels = np.append(all_pred_labels, output.argmax(dim=1, keepdim=True).detach().cpu().data)
        loss = F.cross_entropy(output, label.to(device))
        
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    print("End train")
    sys.stdout.flush()
    loss_epoch_train = np.mean(losses)
    accuracy_epoch_train = metrics.accuracy_score(all_target_labels, all_pred_labels)
    print("Training loss: {}, Training accuracy: {} ".format(loss_epoch_train, accuracy_epoch_train))


# ----------------------------------------------------------------------------------------------------------------
def test_one_epoch(model,
                   data_loader,
                   epoch,
                   device,
                   num_classes):
    
    model = model.to(device)
    model.eval()

    pred_labels = None
    orig_labels = None
    
    with torch.no_grad():
        print("Start test")
        sys.stdout.flush()
        for image_batch, label_batch, _ in data_loader:
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)
            
            output_batch = model(image_batch)
            _, labels_predicted = output_batch.max(1)


            # test accuracy
            if pred_labels is None:
                pred_labels = labels_predicted
                orig_labels = label_batch
            else:
                pred_labels = torch.cat((pred_labels, labels_predicted), dim=0)
                orig_labels = torch.cat((orig_labels, label_batch))
            
    
    # print("Average test loss(weighted): {: .4f}, test accuracy: {: .4f}".format(loss_test, acc))

    # Count occurrences of each class in the ground truth
    # orig_count = Counter(orig_labels.cpu().numpy())

    # Count how many times each class was correctly predicted
    # correct_preds = (pred_labels == orig_labels).cpu().numpy()
    # correct_count = Counter(orig_labels[correct_preds].cpu().numpy())

    # Calculate the fraction of correct predictions for each class
    # fractions = {class_id: f"{correct_count[class_id]} / {orig_count[class_id]}" if orig_count[class_id] > 0 else 0
    #             for class_id in range(num_classes)}

    # Print the fraction of correct predictions for each class
    # print("Fraction of correct predictions per class:")
    # for class_id, fraction in fractions.items():
    #     print(f"Class {class_id}: {fraction}")
    
    # Calculate per-class Accuracies
    accuracy = Accuracy(task="multiclass", num_classes=num_classes, average=None).to(device)
    accuracy_val = accuracy(pred_labels, orig_labels)
    accuracy_results = list([t.item() for t in accuracy_val])
    accuracy_avg = Accuracy(task="multiclass", num_classes=num_classes, average='micro').to(device)
    accuracy_val = accuracy_avg(pred_labels, orig_labels).item()
    accuracy_results.append(accuracy_val)

    print("New Accuracy")
    print(accuracy_results)
    print("End test")
    sys.stdout.flush()


# ----------------------------------------------------------------------------------------------------------------
# def test_one_epoch_temp(model,
#                    data_loader,
#                    epoch,
#                    device,
#                    num_classes):
#     test_metrics = {'test_acc': [], 'test_lss': [], 'cmatrix': [], 'f1': [], 'auc': [], 'fpr': [], 'tpr': [], 'thr': []}

#     loss_test = 0
#     n_test_samples = 0

#     correct = 0
#     total = 0

#     all_labels = []
#     all_probs = []
    
#     model = model.to(device)
#     model.eval()

#     pred_labels = None
#     pred_probs = None
#     orig_labels = None
    
#     with torch.no_grad():
#         print("Start test")
#         sys.stdout.flush()
#         for image, label, _ in data_loader:
#             label_batch = label.to(device).flatten()
#             image, label = image.detach().to(device), label.detach().to(device)
#             output = model(image)
#             loss_fn = nn.CrossEntropyLoss()
#             n_test_samples += len(label)
#             loss_test = loss_test + (len(label) * (loss_fn(output, label).item()))

#             _, pred = torch.max(output.data, 1)
#             _, pred_alt = output.max(1)

#             # test accuracy
#             if pred_labels is None:
#                 pred_labels = pred_alt
#                 orig_labels = label_batch
#                 pred_probs = output
#             else:
#                 pred_labels = torch.cat((pred_labels, pred_alt), dim=0)
#                 orig_labels = torch.cat((orig_labels, label_batch))
#                 pred_probs = torch.cat((pred_probs, output))

#             total += label.size(0)
#             correct += (pred.cpu() == label.cpu()).sum()

#             # Store all labels and predictions for F1 and AUC computation
#             all_labels.extend(label.cpu().numpy())
#             all_probs.extend(output.cpu().detach().numpy())

            
    
#     loss_test /= n_test_samples
#     test_metrics['test_lss'].append(loss_test)

#     # Convert lists to numpy arrays for sklearn metrics
#     all_labels = np.array(all_labels)
#     all_probs = np.array(all_probs)

#     # Overall accuracy
#     acc = 100*float(correct)/float(total)

    
#     print("Average test loss(weighted): {: .4f}, test accuracy: {: .4f}".format(loss_test, acc))


#     # test_metrics['class_auc'] = class_aucs

#     # Compute average ROC curve
#     mean_fpr = np.linspace(0, 1, 100)
#     mean_tpr = np.zeros_like(mean_fpr)
#     for i in range(num_classes):
#         fpr, tpr, _ = roc_curve((all_labels == i).astype(int), all_probs[:, i])
#         mean_tpr += np.interp(mean_fpr, fpr, tpr)
    
#     mean_tpr /= num_classes

#     # Print the metrics
#     print("Average test loss (weighted): {:.4f}".format(loss_test))
    
#     # Calculate per-class Accuracies
#     accuracy = Accuracy(task="multiclass", num_classes=num_classes, average=None).to(device)
#     accuracy_val = accuracy(pred_labels, orig_labels)
#     accuracy_results = list([t.item() for t in accuracy_val])
#     accuracy_avg = Accuracy(task="multiclass", num_classes=num_classes, average='micro').to(device)
#     accuracy_val = accuracy_avg(pred_labels, orig_labels).item()
#     accuracy_results.append(accuracy_val)

#     # Calculate per-class F1-scores
#     f1 = F1Score(task="multiclass", num_classes=num_classes, average=None).to(device)
#     f1_val = f1(pred_labels, orig_labels)
#     f1_results = list([t.item() for t in f1_val])

#     # Calculate micro-average F1-score
#     f1_avg = F1Score(task="multiclass", num_classes=num_classes, average='micro').to(device)
#     f1_micro = f1_avg(pred_labels, orig_labels).item()
#     f1_results.append(f1_micro)
    
#     # Calculate per-class AUC scores
#     aucroc = AUROC(task="multiclass", num_classes=num_classes, average=None).to(device)
#     auc_val = aucroc(pred_probs, orig_labels)
#     auc_results = list([t.item() for t in auc_val])

#     # Calculate micro-average AUC score
#     auc_avg = AUROC(task="multiclass", num_classes=num_classes, average='weighted').to(device)
#     auc_micro = auc_avg(pred_probs, orig_labels).item()
#     auc_results.append(auc_micro)

#     print("New Accuracy")
#     print(accuracy_results)
#     print("New F1")
#     print(f1_results)
#     print("New AUC")
#     print(auc_results)
#     print("End test")
#     sys.stdout.flush()
    
#     return test_metrics
