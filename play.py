from model import *
from torch.multiprocessing import Pipe
import mario_env
import torch
class Player:
    def __init__(self,load_path):
      
       
        checkpoint = torch.load(load_path)
        torch.cuda.empty_cache()
        flag.SHOW_GAME=True

        print("loaded model weigths from checkpoint")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model(num_action=7).to(self.device)
        self.model.load_state_dict(checkpoint['new_model_state_dict'])


        self.model.eval()
    def play(self):
        parent, child= Pipe()
        env= mario_env.MarioEnv(0,child,1,0)
        env.start()
        self.current_observation=np.zeros((4,84,84))
        while True:
            observation_tensor = torch.from_numpy(np.expand_dims(self.current_observation,0)).float().to(self.device)

            predicted_action, value1,value2 = self.model.step(observation_tensor)
            print("action choosen is",predicted_action)
            # self.current_observation,rew,info,done=self.env.step(predicted_action)
            parent.send(predicted_action[0])
            self.current_observation,rew,done= parent.recv()
            print("rewards is",rew)




