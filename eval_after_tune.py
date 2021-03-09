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
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter


def main(args):
    # test_dictionary = '/share/contrastive_learning/data/sup_data/data_0122/val_patch'
    # test_dictionary = '/share/contrastive_learning/data/crop_after_process_doctor/crop_test_screened-20210207T180715Z-001/crop_test_screened'
    test_dictionary = '/share/contrastive_learning/data/crop_after_process_doctor/crop_train_for_exp/crop_test_screened'
    test_loader = torch.utils.data.DataLoader(
        # dataset=get_dataset(
        #     transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs),
        #     train=False,
        #     **args.dataset_kwargs
        # ),
        dataset=datasets.ImageFolder(root=test_dictionary,
                                     transform=get_aug(train=False, train_classifier=False, **args.aug_kwargs)),
        batch_size=args.eval.batch_size,
        shuffle=False,
        **args.dataloader_kwargs
    )

    model = get_backbone(args.model.backbone)
    # classifier = nn.Linear(in_features=model.output_dim, out_features=16, bias=True).to(args.device)

    # assert args.eval_from is not None
    # save_dict = torch.load(args.eval_from, map_location='cpu')
    # msg = model.load_state_dict({k[9:]: v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')},
    #                             strict=True)
    # for ep in range(100):
    MODEL = '/share/contrastive_learning/SimSiam_PatrickHua/SimSiam-main-v2/SimSiam-main/checkpoint/exp_0206_eval/99_all_new/simsiam-TCGA-0126-128by128_tuneall_36.pth'

    # Load the model for testing
    # model = get_backbone(args.model.backbone)
    # model = model.cuda()
    # model = torch.nn.DataParallel(model)
    model = torch.load(MODEL)
    # model = model.load_state_dict({k[9:]: v for k, v in dict.items() if k.startswith('backbone.')},
    # strict=True)
    # model = model.load_state_dict(torch.load(MODEL))
    # model = model.load_state_dict({k: v for k, v in torch.load(MODEL).items() if k.startswith('module.')})
    model.eval()

    # print(msg)
    # if torch.cuda.device_count() > 1: classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
    # classifier = torch.nn.DataParallel(classifier)
    # define optimizer
    optimizer = get_optimizer(
        args.eval.optimizer.name, model,
        lr=args.eval.base_lr * args.eval.batch_size / 256,
        momentum=args.eval.optimizer.momentum,
        weight_decay=args.eval.optimizer.weight_decay)


    acc_meter = AverageMeter(name='Accuracy')

    # Start training
    acc_meter.reset()

    pred_label_for_f1 = np.array([])
    true_label_for_f1 = np.array([])
    for idx, (images, labels) in enumerate(test_loader):
        with torch.no_grad():
            feature = model(images.to(args.device))
            preds = feature.argmax(dim=1)
            correct = (preds == labels.to(args.device)).sum().item()

            preds_arr = preds.cpu().detach().numpy()
            labels_arr = labels.cpu().detach().numpy()
            pred_label_for_f1 = np.concatenate([pred_label_for_f1, preds_arr])
            true_label_for_f1 = np.concatenate([true_label_for_f1, labels_arr])
            acc_meter.update(correct / preds.shape[0])

    f1 = f1_score(true_label_for_f1, pred_label_for_f1, average='macro')
    # precision = precision_score(true_label_for_f1, pred_label_for_f1, average='micro')
    # recall = recall_score(true_label_for_f1, pred_label_for_f1, average='micro')

    print('Epoch : ', str(36),  f'Accuracy = {acc_meter.avg * 100:.2f}', 'F1 score =  ', f1)
    print('F1 score =  ', f1)
    cm = confusion_matrix(true_label_for_f1, pred_label_for_f1)
    np.savetxt("foo_36.csv", cm, delimiter=",")

if __name__ == "__main__":
    main(args=get_args())
