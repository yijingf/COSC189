import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import AugmentedConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.set_default_dtype(torch.float32)

def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

class temporal(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.8, roi='pSTG'):
        super(temporal, self).__init__()
        
        print("Start: Build.")
        stage_3 = {'pSTG': 5080, 'HG': 3440, 'aSTG': 2280}
#         stage_3 = {'pSTG': 5080, 'HG': 3440, 'aSTG': 2280}
        n_Stages = [5, 10, 20, stage_3[roi], 256]
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, n_Stages[0], kernel_size=(1,3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(n_Stages[0]),
            nn.ReLU(),
            nn.MaxPool2d((4, 1), stride=(4, 1))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(n_Stages[0], n_Stages[1], kernel_size=(1,3), stride=1, padding=(0, 1)),
            nn.BatchNorm2d(n_Stages[1]),
            nn.ReLU(),
            nn.MaxPool2d((4, 1), stride=(4, 1))
        )
    
        self.aconv1 = AugmentedConv(in_channels=n_Stages[1], out_channels=n_Stages[2], 
                                    kernel_size=2, dk=12, dv=3, Nh=3, relative=False, stride=2).to(device)
        
        self.bn1 = nn.BatchNorm2d(n_Stages[2], momentum=0.9)
        self.linear1 = nn.Linear(n_Stages[3], n_Stages[4])
        self.linear2 = nn.Linear(n_Stages[4], num_classes)
        self.apply(_weights_init)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.aconv1(out)
        out = F.avg_pool2d(out, (8,1))
        out = self.bn1(out)
        out = out.view(out.size(0), -1)
#         out = F.relu(self.linear1(out))
        out = self.linear1(out)
        out = self.linear2(out)
        
        return out
