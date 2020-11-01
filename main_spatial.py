import resnet as ResNet
from utils import SpatialDataset, calculate_accuracy, AverageMeter

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models

import numpy as np
import os
import argparse

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cpu" if not torch.cuda.is_available() else "cuda:0"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--data_mode", type=str, default='') # _runs_out_align,
    parser.add_argument("--ckpt", type=str, default="") # ./models/spatial_model
    return parser.parse_args()


def load_pretrained_model_utils(model_path):
    ckpt = torch.load(model_path)
    resnet = ResNet.generate_model(model_depth=18, n_classes=700)
    resnet.load_state_dict(ckpt['state_dict'])
    return resnet


def eval_model(model, data):
    model.eval()
    val_accuracies = AverageMeter()
    confusion_mat = np.zeros((5, 5))
    with torch.no_grad():
        for i, (input_data, labels) in enumerate(data):
            input_data = input_data.to(device)
            labels = labels.flatten().to(device)

            outputs = model(input_data) # bs X 5
            acc = calculate_accuracy(outputs, labels)

            val_accuracies.update(acc, input_data.shape[0])
            if args.mode == 'test':
                _, pred = outputs.topk(1, 1, largest=True, sorted=True)
                pred = pred.flatten().cpu()
                labels = labels.cpu()
                for pre, label in zip(pred, labels):
                    confusion_mat[int(label)][int(pre)] += 1
    if args.mode == 'test':
        print(confusion_mat)
    return val_accuracies


def train(model, train_data, val_data, criterion, optimizer, tb_writer, ckpt_saved_path, save_name):
    best_acc = -float('inf')

    for ind_epo in range(1, 50):
        losses = AverageMeter()
        accuracies = AverageMeter()
        val_accuracies = AverageMeter()
        model.train()
        for i, (input_data, labels) in enumerate(train_data):
            input_data = input_data.to(device)
            labels = labels.flatten().to(device)

            outputs = model(input_data)
            loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.cpu().item(), input_data.shape[0])
            accuracies.update(acc, input_data.shape[0])
            if i % (len(train_data) // 10) == 0:
                print(f'Epoch: [{ind_epo}][{i + 1}/{len(train_data)}]\t' +
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(loss=losses,
                                                                 acc=accuracies))
        val_accuracies = eval_model(model, val_data)

        print(f'Epoch: {ind_epo} \t Acc:{round(val_accuracies.avg, 3)}')
        if best_acc < val_accuracies.avg:
            best_acc = val_accuracies.avg
            torch.save({'model':model.state_dict(),
                        'epoch':ind_epo,
                        'optimizer':optimizer},
                       os.path.join(ckpt_saved_path, f"best_{save_name}.pt"))

        tb_writer.add_scalar('train/loss', losses.avg, ind_epo)
        tb_writer.add_scalar('train/acc', accuracies.avg, ind_epo)
        tb_writer.add_scalar('val/acc', val_accuracies.avg, ind_epo)


def test(model, data):
    val_accuracies = eval_model(model, data)
    print(f'Test Acc:{round(val_accuracies.avg, 3)}')


def main_3d():
    pretrained_path = "./models/r3d18_K_200ep.pth"

    resnet = load_pretrained_model_utils(pretrained_path)

    # random_data = torch.rand((1, 3, 10, 256, 256))
    # out = resnet(random_data)
    # print(out.shape)

    # reset vital layers
    resnet.conv1 = torch.nn.Conv3d(48,
                                   64,
                                   kernel_size=(7, 7, 7),
                                   stride=(1, 2, 2),
                                   padding=(3, 3, 3),
                                   bias=False) # original mistake
    resnet.fc = torch.nn.Linear(resnet.fc_in, 5)
    resnet.to(device)

    optimizer = torch.optim.SGD(resnet.parameters(),
                                lr=1e-3,
                                momentum=0.9,
                                weight_decay=1e-3)

    interval_size = args.interval
    criterion = torch.nn.CrossEntropyLoss().to(device)
    if args.mode == 'train':
        appended_name = f'_bs{args.batch_size}_interval_{interval_size}{args.data_mode}_template' # transform or not
        train_data_loader = DataLoader(SpatialDataset(mode=f'train{args.data_mode}', max_interval_size=interval_size,
                                                      transform=False),
                                       batch_size=args.batch_size, shuffle=True)
        val_data_loader = DataLoader(SpatialDataset(mode=f'val{args.data_mode}', max_interval_size=interval_size,
                                                    transform=False),
                                     batch_size=args.batch_size, shuffle=False)
        tb_writer = SummaryWriter(log_dir="./log", filename_suffix=appended_name)

        train(resnet,
              train_data=train_data_loader,
              val_data=val_data_loader,
              criterion=criterion,
              optimizer=optimizer,
              tb_writer=tb_writer,
              ckpt_saved_path=os.path.join('models', 'spatial_model'),
              save_name=appended_name
              )

    elif args.mode == 'test':
        if args.ckpt == "" or not os.path.exists(args.ckpt):
            raise Exception("Need specify which ckpt should be loaded")
        ckpt = torch.load(args.ckpt)
        resnet.load_state_dict(ckpt['model'])
        test_data_loader = DataLoader(SpatialDataset(mode=f'test{args.data_mode}', max_interval_size=interval_size,
                                                     transform=True), # todo check if need transform
                                      batch_size=args.batch_size, shuffle=False)
        test(resnet, test_data_loader)


def main_2d():
    resnet = models.resnet18(pretrained=True)

    # reset vital layers
    resnet.conv1 = torch.nn.Conv2d(48,
                                   64,
                                   kernel_size=(7, 7),
                                   stride=(2, 2),
                                   padding=(3, 3),
                                   bias=False)
    resnet.fc = torch.nn.Linear(512, 5)

    # random_data = torch.rand((1, 48, 132, 176))
    # out = resnet(random_data)
    # print(out.shape)

    resnet.to(device)
    optimizer = torch.optim.SGD(resnet.parameters(),
                                lr=1e-3,
                                momentum=0.9,
                                weight_decay=1e-3)
    #
    interval_size = args.interval
    criterion = torch.nn.CrossEntropyLoss().to(device)
    if args.mode == 'train':
        appended_name = f'_bs{args.batch_size}_{args.data_mode}_transform_template_2d' # transform or not
        train_data_loader = DataLoader(SpatialDataset(mode=f'train{args.data_mode}', max_interval_size=1,
                                                      transform=True),
                                       batch_size=args.batch_size, shuffle=True)
        val_data_loader = DataLoader(SpatialDataset(mode=f'val{args.data_mode}', max_interval_size=1,
                                                    transform=True),
                                     batch_size=args.batch_size, shuffle=False)
        tb_writer = SummaryWriter(log_dir="./log", filename_suffix=appended_name)

        train(resnet,
              train_data=train_data_loader,
              val_data=val_data_loader,
              criterion=criterion,
              optimizer=optimizer,
              tb_writer=tb_writer,
              ckpt_saved_path=os.path.join('models', 'spatial_model'),
              save_name=appended_name
              )

    elif args.mode == 'test':
        if args.ckpt == "" or not os.path.exists(args.ckpt):
            raise Exception("Need specify which ckpt should be loaded")
        ckpt = torch.load(args.ckpt)
        resnet.load_state_dict(ckpt['model'])
        test_data_loader = DataLoader(SpatialDataset(mode=f'test{args.data_mode}', max_interval_size=1,
                                                     transform=True),
                                      batch_size=args.batch_size, shuffle=False)
        test(resnet, test_data_loader)


if __name__ == '__main__':
    args = get_args()
    # main_3d()
    main_2d()
