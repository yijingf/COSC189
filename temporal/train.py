import torch
import torch.nn as nn
import torch.optim as optim

from DataLoader import data_loader
from temporal_model import temporal

import argparse
from tqdm import tqdm
import time
import os

torch.set_default_dtype(torch.float32)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# Model config
learning_rate = 1e-3
batch_size = 8
epochs = 100

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(model, train_loader, optimizer, epoch):
    
    criterion = nn.CrossEntropyLoss().to(device)
    
    model.train()
    step = 0
    train_loss = 0
    train_acc = 0
    for data, target in tqdm(train_loader, desc="epoch " + str(epoch), mininterval=1):
        adjust_learning_rate(optimizer, epoch)
        data, target = data.to(device), target.to(device)
        
#         target = torch.nn.functional.one_hot(target)
        
        optimizer.zero_grad()
        output = model(data)
        
#         print(output.shape, target.shape)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.data
        y_pred = output.data.max(1)[1]

        acc = float(y_pred.eq(target.data).sum()) / len(data) * 100.
        train_acc += acc
        step += 1
        if step % 10 == 0:
            print_out = "[Epoch {0:4d}] Loss: {1:2.3f} Acc: {2:.3f}%".format(epoch, loss.data, acc)
            print(print_out, end='')
#             with open("{}_log.txt".format(roi), "a") as f:
#                 f.write(print_out + '\n')
            for param_group in optimizer.param_groups:
                print(",  Current learning rate is: {}".format(param_group['lr']))

    length = len(train_loader.dataset) // batch_size
    return train_loss / length, train_acc / length


def get_test(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="evaluation", mininterval=1):
            data, target = data.to(device), target.to(device)
            output = model(data)
            prediction = output.data.max(1)[1]
            correct += prediction.eq(target.data).sum()

    acc = 100. * float(correct) / len(test_loader.dataset)
    return acc

def main(roi, leave_out, load_pretrained=False):

    result_dir = os.path.join(leave_out, roi)
    checkpoint_dir = os.path.join(result_dir, 'checkpoint')
    reporting_dir = os.path.join(result_dir, 'reporting')

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(reporting_dir, exist_ok=True)

    checkpoint_fname = os.path.join(checkpoint_dir, 'best_model_ckpt.t7')
    reporting_fname = os.path.join(reporting_dir, 'best_model.txt')
    

    print("Loading training data.")
    train_loader = data_loader(batch_size, roi=roi, mode='train', leave_out=leave_out, shuffle=True)
    print("Loading validation data.")
    val_loader = data_loader(batch_size, roi=roi, mode='val', leave_out=leave_out, shuffle=False)
    
    num_classes = 5
    
    model = temporal(num_classes, roi=roi).to(device)
    if load_pretrained:
        filename = "best_model_"
        checkpoint = torch.load(checkpoint_fname)
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        acc = checkpoint['acc']
        max_val_acc = acc
        print("Load Model Accuracy: ", acc, "Load Model end epoch: ", epoch)
    else:
        epoch = 1
        max_val_acc = 0
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=5e-4, momentum=0.9)

    start_time = time.time()

    for epoch in range(epoch, epochs):
        train(model, train_loader, optimizer, epoch)
        val_acc = get_test(model, val_loader)
        if max_val_acc < val_acc:
            print('Saving..')
            state = {
                'model': model.state_dict(),
                'acc': val_acc,
                'epoch': epoch,
            }
   
            torch.save(state, checkpoint_fname)
            max_val_acc = val_acc

        time_interval = time.time() - start_time
        time_split = time.gmtime(time_interval)
        print("Training time: ", time_interval, "Hour: ", time_split.tm_hour, "Minute: ", time_split.tm_min, "Second: ", time_split.tm_sec)
        print("Validation acc:", max_val_acc, "time: ", time.time() - start_time)

        with open(reporting_fname, "a") as f:
            f.write("Epoch: " + str(epoch) + " " + "Best acc: " + str(max_val_acc) + "\n")
            f.write("Training time: " + str(time_interval) + "Hour: " + str(time_split.tm_hour) + "Minute: " + str(
                time_split.tm_min) + "Second: " + str(time_split.tm_sec))
            f.write("\n")
            

if __name__ == "__main__":
    # roi = 'HG' #'aSTG', 'pSTG'
    parser = argparse.ArgumentParser("parameters")

    parser.add_argument("--roi", type=str, default="HG")
    parser.add_argument("-l", type=str, default='r', help="r for run_out, others for sub_out")
    parser.add_argument("--load-pretrained", type=bool, default=False)

    args = parser.parse_args()
    
    if args.l == 'r':
        leave_out='run_out'
    else:
        leave_out = 'sub_out'
    
#     print(args.roi, leave_out, args.load_pretrained)
    main(roi=args.roi, leave_out=leave_out, load_pretrained=args.load_pretrained)