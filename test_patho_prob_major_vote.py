import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
# from nets import *
import time, os, copy, argparse
import multiprocessing
from matplotlib import pyplot as plt
# from model import *
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
from torchvision.models import resnet50, resnet18

valid_directory = '../data/test/'

# Batch size
bs = 64
# Number of classes
num_classes = 9
# Number of workers
num_cpu = multiprocessing.cpu_count()
# num_cpu = 0

# Applying transforms to the data
image_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(size=128, scale=(0.8, 1.0)),
        transforms.Resize(size=128),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=128),
        # transforms.Resize(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# valid_dataset1 = datasets.ImageFolder(root=valid_directory + '4C83', transform=image_transforms['valid'])
valid_dataset2 = datasets.ImageFolder(root=valid_directory + 'EB70', transform=image_transforms['valid'])

# Load data from folders
dataset = {
    'valid': data.ConcatDataset([valid_dataset2])
}

# Size of train and validation data
dataset_sizes = {
    'valid': len(dataset['valid'])
}

# Create iterators for data loading
dataloaders = {
    'valid': data.DataLoader(dataset['valid'], batch_size=bs, shuffle=True,
                             num_workers=num_cpu, pin_memory=True, drop_last=True)
}

# Print the train and validation data sizes
print("\nValidation-set size:", dataset_sizes['valid'])

# Set default device as gpu, if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL = '../model/simsiam-TCGA-0218-nearby_0223173249.pth'
CLASSI0 = '../model/classifier/D8E6_tunelinear_3.pth'
CLASSI1 = '../model/classifier/117E_tunelinear_10.pth'
CLASSI2 = '../model/classifier/676F_tunelinear_1.pth'
CLASSI3 = '../model/classifier/E2D7_tunelinear_4.pth'
CLASSI4 = '../model/classifier/BE52_tunelinear_1.pth'

# Load the model for testing
backbone = 'resnet50'
backbone = eval(f"{backbone}()")
backbone.output_dim = backbone.fc.in_features
backbone.fc = torch.nn.Identity()
model = backbone
save_dict = torch.load(MODEL, map_location='cpu')
model.load_state_dict({k[9:]: v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')},
                            strict=True)
# model = torch.load(MODEL)
model.eval()
model = model.to(device)

classifier0 = torch.load(CLASSI0)
classifier0.eval()
classifier0 = classifier0.to(device)

classifier1 = torch.load(CLASSI1)
classifier1.eval()
classifier1 = classifier1.to(device)

classifier2 = torch.load(CLASSI2)
classifier2.eval()
classifier2 = classifier2.to(device)

classifier3 = torch.load(CLASSI3)
classifier3.eval()
classifier3 = classifier3.to(device)

classifier4 = torch.load(CLASSI4)
classifier4.eval()
classifier4 = classifier4.to(device)

since = time.time()
best_acc = 0.0

model.eval()  # Set model to evaluate mode
running_corrects = 0

pred = []
true = []
for inputs, labels in dataloaders['valid']:
    inputs = inputs.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    # forward
    # outputs = model(inputs)
    # _, preds = torch.max(outputs, 1)
    preds = np.zeros(bs)
    feature = model(inputs)
    out_prob0 = np.array(classifier0(feature).detach().cpu())
    out_prob1 = np.array(classifier1(feature).detach().cpu())
    out_prob2 = np.array(classifier2(feature).detach().cpu())
    out_prob3 = np.array(classifier3(feature).detach().cpu())
    out_prob4 = np.array(classifier4(feature).detach().cpu())

    for pred_i in range(bs):
        norm_prob_0 = preprocessing.normalize([out_prob0[pred_i]])
        norm_prob_1 = preprocessing.normalize([out_prob1[pred_i]])
        norm_prob_2 = preprocessing.normalize([out_prob2[pred_i]])
        norm_prob_3 = preprocessing.normalize([out_prob3[pred_i]])
        norm_prob_4 = preprocessing.normalize([out_prob4[pred_i]])
        norm_prob_all = norm_prob_0 + norm_prob_1 + norm_prob_2 + norm_prob_3 + norm_prob_4
        preds[pred_i] = np.argmax(norm_prob_all)

    preds_list = list(preds)
    labels_list = list(np.array(labels.cpu()))
    pred.append(preds_list)
    true.append(labels_list)

pred = sum(pred, [])
true = sum(true, [])
# epoch_acc = running_corrects.double() / dataset_sizes['valid']
cm = confusion_matrix(true, pred)
f1 = f1_score(true, pred, average='macro')
print('f1 score:  ', f1)
# np.savetxt("cm_0221_triple_200.csv", cm, delimiter=",")
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
balance_acc = balanced_accuracy_score(true, pred)
print('Best balance Acc: {:4f}'.format(balance_acc))

'''
Sample run: python train.py --mode=finetue
'''
