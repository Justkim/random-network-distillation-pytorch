from model import *
class Player:
    def __init__(self,env,load_path):
        self.env=env
        checkpoint = torch.load(load_path)
        torch.cuda.empty_cache()

        print("loaded model weigths from checkpoint")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_observation=self.env.reset()
        self.model = Model(num_action=5).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
    def play(self):

        while True:
            observation_tensor = torch.from_numpy(np.expand_dims(self.current_observation,0)).float().to(self.device)
            predicted_action, value = self.model.step(observation_tensor)
            print("action choosen is",predicted_action)
            self.current_observation,rew,info,done=self.env.step(predicted_action)
            print("rewards is",rew)
            if flag.SHOW_GAME:
                self.env.render()



