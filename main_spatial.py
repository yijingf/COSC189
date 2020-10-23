import torch
import numpy as np
import os
import resnet as ResNet

device = "cpu" if torch.cuda.is_available() else "cuda:0"


def load_pretrained_model(model_path):
    ckpt = torch.load(model_path)
    resnet = ResNet.generate_model(model_depth=18, n_classes=700)
    resnet.load_state_dict(ckpt['state_dict'])
    return resnet


def main():
    pretrained_path = "./models/r3d18_K_200ep.pth"
    resnet = load_pretrained_model(pretrained_path)

    # random_data = torch.rand((1, 3, 10, 256, 256))
    # out = resnet(random_data)
    # print(out.shape)

    # reset vital layers
    resnet.conv1 = torch.nn.Conv3d(36,
                                   64,
                                   kernel_size=(7, 7, 7),
                                   stride=(1, 2, 2),
                                   padding=(3, 3, 3),
                                   bias=False)
    resnet.fc = torch.nn.Linear(resnet.fc_in, 5)


if __name__ == '__main__':
    main()