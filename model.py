import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import flag

class Model(nn.Module):
    def __init__(self,num_action):
        super(Model,self).__init__()

        self.num_action=num_action

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4) #check this input di
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1=nn.Linear(7*7*64,512)
        self.value= nn.Linear(512,1)
        self.policy= nn.Linear(512,self.num_action)
        self.softmax= nn.Softmax()

    def forward_pass(self,input_observations):
        x=F.relu(self.conv1(input_observations))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x=  x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        predicted_value = self.value(x)[:,0]
        policy = self.policy(x)
        return policy,predicted_value

    def step(self,observations):
        policy_tensor, predicted_value_tensor=self.forward_pass(observations)
        softmax_policy_tensor = self.softmax(policy_tensor)
        softmax_policy = softmax_policy_tensor.detach().cpu().numpy()
        predicted_value=predicted_value_tensor.detach().cpu().numpy()
        randoms = np.expand_dims(np.random.rand(softmax_policy.shape[0]), axis=1)
        action = (softmax_policy.cumsum(axis=1) > randoms).argmax(axis=1)
        return action,predicted_value











