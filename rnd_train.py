from rnd_model import TargetModel,PredictorModel
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class RNDTrain:
    def __init__(self,predictor_learning_rate):

        self.losses=[]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_model = TargetModel().to(self.device)
        self.predictor_model = PredictorModel().to(self.device)
        self.optimizer = optim.Adam(self.predictor_model.parameters(), lr=self.learning_rate)
        self.mse_loss = nn.MSELoss()


    def get_intrinsic_rewards(self,input_observation):

        target_value=self.target_model.forward_pass(input_observation)
        predictor_value=self.predictor_model.forward_pass(input_observation)
        intrinsic_reward=self.mse_loss(predictor_value,target_value)
        self.losses.append(intrinsic_reward)
        return intrinsic_reward #check this

    def grad(self):
        with tf.GradientTape() as tape:
            loss=np.array(self.losses)
        gradients = tape.gradient(loss, self.predictor_model.trainable_variables)
        gradients = [(tf.clip_by_value(grad, -0.5, 0.5))
                     for grad in gradients]
        self.losses=[]
        return loss, gradients


    def train_predictor(self):
        loss,grads=self.grad()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.new_model.parameters(), 0.5)
        self.optimizer.step()