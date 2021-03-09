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
from sklearn.metrics import f1_score, balanced_accuracy_score
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import ConcatDataset
import torch.utils.data as data
import csv


def main(args):
    train_info = []
    best_epoch = np.zeros(5)
    for val_folder_index in range(5):
        best_balance_acc = 0
        whole_train_list = ['D8E6', '117E', '676F', 'E2D7', 'BE52']
        val_WSI_list = whole_train_list[val_folder_index]
        train_WSI_list = whole_train_list
        train_WSI_list.pop(val_folder_index)
        train_directory = '../data/finetune/1percent/'
        valid_directory = '../data/finetune/1percent'
        dataset = {}
        dataset_train0 = datasets.ImageFolder(root=train_directory + train_WSI_list[0], transform=get_aug(train=False, train_classifier=True, **args.aug_kwargs))
        dataset_train1 = datasets.ImageFolder(root=train_directory + train_WSI_list[1], transform=get_aug(train=False, train_classifier=True, **args.aug_kwargs))
        dataset_train2 = datasets.ImageFolder(root=train_directory + train_WSI_list[2], transform=get_aug(train=False, train_classifier=True, **args.aug_kwargs))
        dataset_train3 = datasets.ImageFolder(root=train_directory + train_WSI_list[3], transform=get_aug(train=False, train_classifier=True, **args.aug_kwargs))
        dataset['valid'] = datasets.ImageFolder(root=valid_directory + val_WSI_list, transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs))
        dataset['train'] = data.ConcatDataset([dataset_train0, dataset_train1, dataset_train2, dataset_train3])


        train_loader = torch.utils.data.DataLoader(
            dataset=dataset['train'],
            batch_size=args.eval.batch_size,
            shuffle=True,
            **args.dataloader_kwargs
        )
        test_loader = torch.utils.data.DataLoader(
            dataset= dataset['valid'],
            batch_size=args.eval.batch_size,
            shuffle=False,
            **args.dataloader_kwargs
        )

        model = get_backbone(args.model.backbone)
        classifier = nn.Linear(in_features=model.output_dim, out_features=9, bias=True).to(args.device)

        assert args.eval_from is not None
        save_dict = torch.load(args.eval_from, map_location='cpu')
        msg = model.load_state_dict({k[9:]: v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')},
                                    strict=True)

        # print(msg)
        model = model.to(args.device)
        model = torch.nn.DataParallel(model)

        classifier = torch.nn.DataParallel(classifier)
        # define optimizer
        optimizer = get_optimizer(
            args.eval.optimizer.name, classifier,
            lr=args.eval.base_lr * args.eval.batch_size / 256,
            momentum=args.eval.optimizer.momentum,
            weight_decay=args.eval.optimizer.weight_decay)

        # define lr scheduler
        lr_scheduler = LR_Scheduler(
            optimizer,
            args.eval.warmup_epochs, args.eval.warmup_lr * args.eval.batch_size / 256,
            args.eval.num_epochs, args.eval.base_lr * args.eval.batch_size / 256,
                                     args.eval.final_lr * args.eval.batch_size / 256,
            len(train_loader),
        )

        loss_meter = AverageMeter(name='Loss')
        acc_meter = AverageMeter(name='Accuracy')

        # Start training
        global_progress = tqdm(range(0, args.eval.num_epochs), desc=f'Evaluating')
        for epoch in global_progress:
            loss_meter.reset()
            model.eval()
            classifier.train()
            local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.eval.num_epochs}', disable=True)

            for idx, (images, labels) in enumerate(local_progress):
                classifier.zero_grad()
                with torch.no_grad():
                    feature = model(images.to(args.device))

                preds = classifier(feature)

                loss = F.cross_entropy(preds, labels.to(args.device))

                loss.backward()
                optimizer.step()
                loss_meter.update(loss.item())
                lr = lr_scheduler.step()
                local_progress.set_postfix({'lr': lr, "loss": loss_meter.val, 'loss_avg': loss_meter.avg})

            writer.add_scalar('Valid/Loss', loss_meter.avg, epoch)
            writer.add_scalar('Valid/Lr', lr, epoch)
            writer.flush()

            PATH = 'checkpoint/exp_0228_triple_1percent/' + val_WSI_list + '/' + val_WSI_list + '_tunelinear_' + str(epoch) + '.pth'

            torch.save(classifier, PATH)

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
            balance_acc = balanced_accuracy_score(true_label_for_f1, pred_label_for_f1)
            print('Epoch:  ', str(epoch), f'Accuracy = {acc_meter.avg * 100:.2f}')
            print('F1 score =  ', f1, 'balance acc:  ', balance_acc)
            if balance_acc > best_balance_acc:
                best_epoch[val_folder_index] = epoch
                best_balance_acc = balance_acc
            train_info.append([val_WSI_list, epoch, f1, balance_acc])

    with open('checkpoint/exp_0228_triple_1percent/train_info.csv', 'w') as f:
        # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerows(train_info)
    print(best_epoch)

if __name__ == "__main__":
    writer = SummaryWriter()
    main(args=get_args())


