import torch.nn.init as init
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
        self.fc1=nn.Linear(7*7*64,256)
        self.fc2= nn.Linear(256,448)
        self.fc_actor = nn.Linear(448,448)
        self.int_value = nn.Linear(448, 1)
        self.ext_value = nn.Linear(448, 1)
        self.extra = nn.Linear(448, 448)
        self.policy= nn.Linear(448,self.num_action)

        self.softmax= nn.Softmax()

        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                init.orthogonal_(p.weight, np.sqrt(2))
                p.bias.data.zero_()

        init.orthogonal_(self.ext_value.weight, 0.01)
        self.ext_value.bias.data.zero_()

        init.orthogonal_(self.int_value.weight, 0.01)
        self.int_value.bias.data.zero_()

        init.orthogonal_(self.fc_actor.weight, 0.01)
        self.fc_actor.bias.data.zero_()

        init.orthogonal_(self.policy.weight, 0.01)
        self.policy.bias.data.zero_()

        init.orthogonal_(self.extra.weight, 0.1)
        self.extra.bias.data.zero_()

    def forward(self,input_observations):
        x=F.relu(self.conv1(input_observations))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x=  x.view(x.size(0), -1) #flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actor_policy = F.relu(self.fc_actor(x))
        policy = self.policy(actor_policy)
        predicted_int_value= self.int_value(F.relu(self.extra(x))+x)[:,0]
        predicted_ext_value = self.ext_value(F.relu(self.extra(x)) + x) [:,0]
        return policy,predicted_ext_value,predicted_int_value

    def step(self,observations):
        policy_tensor, predicted_ext_value_tensor, predicted_int_value_tensor=self(observations)
        softmax_policy_tensor = F.softmax(input=policy_tensor)
        softmax_policy = softmax_policy_tensor.data.cpu().numpy()
        predicted_ext_value=predicted_ext_value_tensor.data.cpu().numpy()
        predicted_int_value = predicted_int_value_tensor.data.cpu().numpy()
        randoms = np.expand_dims(np.random.rand(softmax_policy.shape[0]), axis=1)
        action = (softmax_policy.cumsum(axis=1) > randoms).argmax(axis=1)
        return action,predicted_ext_value, predicted_int_value











