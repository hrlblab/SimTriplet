import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model, get_backbone
from tools import AverageMeter
from datasets import get_dataset
from optimizers import get_optimizer, LR_Scheduler
from torchvision import datasets
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter


def main(args):
    train_directory = '/share/contrastive_learning/data/sup_data/data_0122/train_patch'
    train_loader = torch.utils.data.DataLoader(
        # dataset=get_dataset(
        #     transform=get_aug(train=False, train_classifier=True, **args.aug_kwargs),
        #     train=True,
        #     **args.dataset_kwargs
        # ),
        dataset=datasets.ImageFolder(root=train_directory, transform=get_aug(train=False, train_classifier=True, **args.aug_kwargs)),
        batch_size=args.eval.batch_size,
        shuffle=True,
        **args.dataloader_kwargs
    )
    test_dictionary = '/share/contrastive_learning/data/sup_data/data_0122/val_patch'
    test_loader = torch.utils.data.DataLoader(
        # dataset=get_dataset(
        #     transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs),
        #     train=False,
        #     **args.dataset_kwargs
        # ),
        dataset=datasets.ImageFolder(root=test_dictionary, transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs)),
        batch_size=args.eval.batch_size,
        shuffle=False,
        **args.dataloader_kwargs
    )

    model = get_backbone(args.model.backbone)
    classifier = nn.Linear(in_features=model.output_dim, out_features=16, bias=True).to(args.device)

    assert args.eval_from is not None
    save_dict = torch.load(args.eval_from, map_location='cpu')
    msg = model.load_state_dict({k[9:]: v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')},
                                strict=True)

    # print(msg)
    model = model.to(args.device)
    model = torch.nn.DataParallel(model)

    # if torch.cuda.device_count() > 1: classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
    classifier = torch.load('checkpoint/simsiam-TCGA-0126-128by128_tunelinear5.pth')


    loss_meter = AverageMeter(name='Loss')
    acc_meter = AverageMeter(name='Accuracy')

    classifier.eval()
    correct, total = 0, 0
    acc_meter.reset()

    pred_label_for_f1 = np.array([])
    true_label_for_f1 = np.array([])
    for idx, (images, labels) in enumerate(test_loader):
        with torch.no_grad():
            feature = model(images.to(args.device))
            preds = classifier(feature).argmax(dim=1)
            correct = (preds == labels.to(args.device)).sum().item()

            preds_arr = preds.cpu().detach().numpy()
            labels_arr = labels.cpu().detach().numpy()
            pred_label_for_f1 = np.concatenate([pred_label_for_f1, preds_arr])
            true_label_for_f1 = np.concatenate([true_label_for_f1, labels_arr])
            acc_meter.update(correct / preds.shape[0])

    f1 = f1_score(true_label_for_f1, pred_label_for_f1, average='macro')
    print(f'Accuracy = {acc_meter.avg * 100:.2f}')
    print('F1 score =  ', f1)

if __name__ == "__main__":
    writer = SummaryWriter()
    main(args=get_args())
















