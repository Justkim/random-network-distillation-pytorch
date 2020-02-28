
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np


class TargetModel(nn.Module):
    def __init__(self,num_action):
        super(TargetModel,self).__init__()

        self.num_action=num_action

        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4) #check this input di
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1=nn.Linear(7*7*64,512)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        for param in self.parameters():
            param.requires_grad = False

    def forward_pass(self,input_observations):
        x=F.leaky_relu(self.conv1(input_observations))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x=  x.view(x.size(0), -1)
        target_value = (self.fc1(x))
        return target_value





class PredictorModel(nn.Module):
    def __init__(self,num_action):
        super(PredictorModel,self).__init__()

        self.num_action=num_action

        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4) #check this input di
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1=nn.Linear(7*7*64,512)
        self.fc2=nn.Linear(512,512)
        self.fc3=nn.Linear(512,512)

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

    def forward_pass(self,input_observations):
        x=F.leaky_relu(self.conv1(input_observations))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x=  x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        predictor_value = self.fc3(x)
        return predictor_value









