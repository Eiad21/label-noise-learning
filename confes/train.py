import sys

import torch
from tqdm import tqdm
import numpy as np
from torch import optim, nn
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR, CosineAnnealingLR

import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc
import sklearn.metrics as metrics
from sklearn.mixture import GaussianMixture

from torchmetrics.classification import Accuracy, F1Score, AUROC, ConfusionMatrix, MulticlassAUROC

torch.cuda.manual_seed(2)
np.random.seed(2)
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
        min_lr = gamma * learning_rate
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

    for _, (image, label, _, _) in enumerate(data_loader):

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
    test_metrics = {'test_acc': [], 'test_lss': [], 'cmatrix': [], 'f1': [], 'auc': [], 'fpr': [], 'tpr': [], 'thr': []}

    loss_test = 0
    n_test_samples = 0

    correct = 0
    total = 0


    all_labels = []
    all_probs = []
    
    model = model.to(device)
    model.eval()

    pred_labels = None
    pred_probs = None
    orig_labels = None

    with torch.no_grad():
        print("Start test")
        sys.stdout.flush()
        for image, label, _, _ in data_loader:
            label_batch = label.to(device).flatten()
            image, label = image.detach().to(device), label.detach().to(device)
            output = model(image)
            loss_fn = nn.CrossEntropyLoss()
            n_test_samples += len(label)
            loss_test = loss_test + (len(label) * (loss_fn(output, label).item()))

            _, pred = torch.max(output.data, 1)
            _, pred_alt = output.max(1)

            # test accuracy
            if pred_labels is None:
                pred_labels = pred_alt
                orig_labels = label_batch
                pred_probs = output
            else:
                pred_labels = torch.cat((pred_labels, pred_alt), dim=0)
                orig_labels = torch.cat((orig_labels, label_batch))
                pred_probs = torch.cat((pred_probs, output))

            total += label.size(0)
            correct += (pred.cpu() == label.cpu()).sum()

            # Store all labels and predictions for F1 and AUC computation
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(output.cpu().detach().numpy())

    
    loss_test /= n_test_samples
    test_metrics['test_lss'].append(loss_test)

    # Convert lists to numpy arrays for sklearn metrics
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Overall accuracy
    acc = 100*float(correct)/float(total)


    test_metrics['test_acc'] = acc
    
    # Compute average ROC curve
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(mean_fpr)
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve((all_labels == i).astype(int), all_probs[:, i])
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
    
    mean_tpr /= num_classes

    # Calculate per-class Accuracies
    accuracy = Accuracy(task="multiclass", num_classes=num_classes, average=None).to(device)
    accuracy_val = accuracy(pred_labels, orig_labels)
    accuracy_results = list([t.item() for t in accuracy_val])
    accuracy_avg = Accuracy(task="multiclass", num_classes=num_classes, average='micro').to(device)
    accuracy_val = accuracy_avg(pred_labels, orig_labels).item()
    accuracy_results.append(accuracy_val)

    # Calculate per-class F1-scores
    f1 = F1Score(task="multiclass", num_classes=num_classes, average=None).to(device)
    f1_val = f1(pred_labels, orig_labels)
    f1_results = list([t.item() for t in f1_val])

    # Calculate micro-average F1-score
    f1_avg = F1Score(task="multiclass", num_classes=num_classes, average='micro').to(device)
    f1_micro = f1_avg(pred_labels, orig_labels).item()
    f1_results.append(f1_micro)
    
    # Calculate per-class AUC scores
    aucroc = AUROC(task="multiclass", num_classes=num_classes, average=None).to(device)
    auc_val = aucroc(pred_probs, orig_labels)
    auc_results = list([t.item() for t in auc_val])

    # Calculate micro-average AUC score
    auc_avg = AUROC(task="multiclass", num_classes=num_classes, average='weighted').to(device)
    auc_micro = auc_avg(pred_probs, orig_labels).item()
    auc_results.append(auc_micro)

    # Print the metrics
    print("Average test loss(weighted): {: .4f}, test accuracy: {: .4f}".format(loss_test, acc))
    print("Class-wise accuracies:\n", accuracy_results)
    print("Class-wise F1 scores:\n", f1_results)
    print("Class-wise AUCs:\n", auc_results)
    # print("Mean FPR: ", mean_fpr)
    # print("Mean TPR: ", mean_tpr)
    
    print("End test")
    sys.stdout.flush()
    
    return test_metrics

# ----------------------------------------------------------------------------------------------------------------
def eval_train(model, epoch, data_loader, sample_size, alpha, num_classes, clustering, device, clean_indices, clean_labels_np):
    label_pred = np.empty(sample_size)
    label_pred[:] = np.nan
    probs = np.empty((sample_size, num_classes))
    probs[:] = np.nan
    per_sample_ce = np.empty(sample_size)
    per_sample_ce[:] = np.nan
    small_ce = []
    large_ce = []
  
    all_clean_labels = np.array([], dtype=np.int64)

    var_output_dataset = np.ones(sample_size).tolist()
    with torch.no_grad():
        print("Start eval")
        sys.stdout.flush()
        for image, label, index, clean in data_loader:
            output = model(image.to(device))
            y_pred = F.softmax(output, dim=1)
                    
            y_pred = torch.clamp(y_pred, 1e-4, 1.0 - 1e-4)
            
            all_clean_labels = np.append(all_clean_labels, clean)
        
            loss_ce = F.cross_entropy(output.cpu(), label, reduction='none')
            var_output_batch = [(float(y_pred[index_local].max() - y_pred[index_local][label[index_local]])) for
                                index_local in range(len(y_pred))]

            for g_i, l_i in zip(index, np.arange(0, len(var_output_batch))):
                var_output_dataset[g_i] = var_output_batch[l_i]
                if clustering != "none":
                    continue
                label_pred[g_i] = output.argmax(dim=1, keepdim=True)[l_i]
                probs[g_i] = y_pred[l_i].cpu().detach()
                per_sample_ce[g_i] = loss_ce[l_i].cpu().detach()
                if var_output_dataset[g_i] <= alpha:
                    small_ce.append(g_i)
                else:
                    large_ce.append(g_i)
        
        if clustering == "gmm":
            gmm = GaussianMixture(n_components=2)
            var_output_dataset_np = np.array(var_output_dataset)
            clusters = gmm.fit_predict(var_output_dataset_np.reshape(-1, 1))

            # Assign samples based on cluster center proximity
            cluster_centers = gmm.means_
            clean_cluster = np.argmin(cluster_centers)

            
            small_ce = np.where(clusters == clean_cluster)[0]
            large_ce = np.where(clusters != clean_cluster)[0]
    
        print("End eval")
        sys.stdout.flush()

        clean_labels_epoch = torch.LongTensor(small_ce)
        noisy_labels_epoch = torch.LongTensor(large_ce)

        return clean_labels_epoch, noisy_labels_epoch, torch.tensor(probs), per_sample_ce

