import numpy as np
import gym
from collections import deque
import cv2
from torch.multiprocessing import Process
import flag
cv2.ocl.setUseOpenCL(False)
import time


class MontezumaRevenge(Process):
    def __init__(self,env_id,child,action_re,p,max_steps):
        super(MontezumaRevenge, self).__init__()
        self.env = self.make_env()
        self.env.reset()
        self.child=child
        self.env_id=env_id
        self.action_re=action_re
        self.p=p
        self.last_action = 0
        self.ep_num = 0
        self.env_id = env_id
        self.steps=0
        self.max_steps=max_steps

    def make_env(self):
        env = gym.make("MontezumaRevengeNoFrameskip-v4")
        env = PreprocessFrame(env)
        return env

    def run(self):
        while True:
            action = self.child.recv()
            reward=0
            if flag.STICKY_ACTION:
                if (np.random.rand() <= self.p):
                    action = self.last_action
                self.last_action = action
            for i in range(0, self.action_re):
                obs,rew, done, info = self.env.step(action)
                reward+=rew
                # if info['ale.lives'] < 6:
                #    done = True
                if self.steps>self.max_steps:
                    done=True
                if done:
                    #print("env: " + str(self.env_id) + " episode: "+ str(self.ep_num))
                    self.ep_num += 1
                    self.steps=0
                    obs = self.env.reset()
                    break

            if flag.SHOW_GAME:
                self.env.render()
                time.sleep(0.5)
            self.steps+=1
            self.child.send([obs, reward, done])



class PreprocessFrame(gym.ObservationWrapper):

    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(self.height, self.width, 1), dtype=np.uint8)
        self.frame_deque = deque([np.zeros((self.height, self.width)), np.zeros((self.height, self.width)),
                                  np.zeros((self.height, self.width)), np.zeros((self.height, self.width))], maxlen=4)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = frame[:, :, None]
        # frame = frame / 255.

        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return self.stack_frames(frame)

    def stack_frames(self, new_frame):
        self.frame_deque.append(new_frame)
        return np.stack(self.frame_deque)


#
# new_env = MontezumaRevenge.make_env(0)
# new_env.reset()
# #
# while True:
#       obs,rew,done,info= new_env.step(new_env.action_space.sample())
#       print("info is", info)
#       print("reward is", rew)
#       print(type(obs))
#       new_env.render()
