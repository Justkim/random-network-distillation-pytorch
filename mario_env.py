import gym
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT,COMPLEX_MOVEMENT
import flag
from collections import deque
import numpy as np
from torch.multiprocessing import Pipe, Process
import cv2
cv2.ocl.setUseOpenCL(False)


class MarioEnv(Process):
    def __init__(self,env_id,child,action_re,p,max_steps):
        super(MarioEnv, self).__init__()
        self.child = child
        self.env = self.make_env(0)
        self.env.reset()
        self.action_re=action_re
        self.p=p
        self.last_action=0
        self.ep_num=0
        self.env_id=env_id
        self.progress_reward=0
        self.max_steps=max_steps
        self.steps=0


    def make_env(self,env_idx):
        """
        Create an environment with some standard wrappers.
        """

        # Make the environment

        levelList = ['SuperMarioBros-v0', 'SuperMarioBros-2-1-v0', 'SuperMarioBros-3-1-v0', 'SuperMarioBros-4-1-v0',
                     'SuperMarioBros-5-1-v0', 'SuperMarioBros-6-1-v0', 'SuperMarioBros-7-1-v0', 'SuperMarioBros-8-1-v0']
        env = gym_super_mario_bros.make(levelList[env_idx])
        if flag.ENV == "complex-mario":
            env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)
        elif flag.ENV == "simple-mario":
            env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)
        else:
            print("env type error: env not recognized")
            exit()

        env = PreprocessFrame(env)
        # env = RewardScaler(env)
        return env

    def run(self):
        while True:
            action= self.child.recv()
            reward=0
            if flag.STICKY_ACTION:
                if(np.random.rand()<=self.p):
                    action=self.last_action
                self.last_action = action
            global progress_reward
            for i in range(0,self.action_re):
                obs,progress_reward,done,info = self.env.step(action)
                if info['flag_get'] == True:
                    rew = 1
                else:
                    rew = 0
                reward+=rew
                if info['life']<2:
                    done=True
                if self.steps > self.max_steps:
                    done = True
                if done:
                    print("env: " + str(self.env_id) + " episode: " + str(self.ep_num) +" progress; "+str(self.progress_reward)+ " max_x: " + str(
                        info['x_pos']))
                    self.ep_num += 1
                    self.progress_reward=0
                    obs = self.env.reset()
                    self.steps=0
                    break

            progress_reward=progress_reward/15
            self.progress_reward+=progress_reward

            if flag.SHOW_GAME:
                self.env.render()
            self.steps += 1
            self.child.send([obs, reward,done])




class PreprocessFrame(gym.ObservationWrapper):
    """
    Here we do the preprocessing part:
    - Set frame to gray
        - Resize the frame to 96x96x1
    """
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.width = 84
        self.height = 84
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)
        self.frame_deque = deque([np.zeros((self.height,self.width)),np.zeros((self.height,self.width)),np.zeros((self.height,self.width)), np.zeros((self.height,self.width))], maxlen=4)


    def observation(self, frame):
        # Set frame to gray
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


        # frame = frame[35: , :,None]


        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        # frame = frame[:, :, None]
        frame=frame/255.0
        # frame=frame/255
        # if flag.DEBUG:
        #     cv2.imshow("frame",frame)
        #     cv2.waitKey(0)

        return self.stack_frames(frame)

    def stack_frames(self,new_frame):
        self.frame_deque.append(new_frame)
        return np.stack(self.frame_deque)


