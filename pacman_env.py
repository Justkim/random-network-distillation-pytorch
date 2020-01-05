# Part taken from adborghi fantastic implementation
# https://github.com/aborghi/retro_contest_agent/blob/master/fastlearner/ppo2ttifrutti_sonic_env.py
import numpy as np
import gym
from collections import deque
import cv2
cv2.ocl.setUseOpenCL(False)

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
        self.frame_deque = deque([np.zeros((self.height, self.width)), np.zeros((self.height, self.width)),
                                  np.zeros((self.height, self.width)), np.zeros((self.height, self.width))], maxlen=4)

    def observation(self, frame):

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = frame[:200, :, None]
        frame= frame/255.

        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        # if flag.DEBUG:
        print(frame.shape)

        return self.stack_frames(frame)

    def stack_frames(self, new_frame):
        self.frame_deque.append(new_frame)
        return np.stack(self.frame_deque)


class RewardScaler(gym.ActionWrapper):
    def __init__(self, env):
        super(RewardScaler, self).__init__(env)

    def step(self, action):
        obs,rew,done,info=self.env.step(action)
        if done:
            if info['ale.lives']==0:
                rew=0
            else:
                rew=1
        else:
            rew=0


def make_env(env_idx):
    """
    Create an environment with some standard wrappers.
    """

    # record_path = "./records/" + dictsMsPacman[env_idx]['state']
    #env = gym_super_mario_bros.make(levelList[env_idx])
    env = gym.make("MsPacman-v0")

    #SuperMarioBros-v0
    #SuperMarioBrosRandomStages
   # env = BinarySpaceToDiscreteSpaceEnv(env,RIGHT_ONLY)

    env = RewardScaler(env)

    # PreprocessFrame
    env = PreprocessFrame(env)

    # Allow back tracking that helps agents are not discouraged too heavily
    # from exploring backwards if there is no way to advance
    # head-on in the level.
  #  env = AllowBacktracking(env)
    return env


def make_train_0():
    new_env=make_env(0)
    # new_env = gym.wrappers.Monitor(new_env, "recording0")
    return new_env


new_env=make_env(0)
new_env.reset()

while True:
    action=input()
    a,b,c,d=new_env.step(int(action))

    print("info is",d)
    print("reward is",b)

    new_env.render()
    if c:
        exit()
#     print("reward is", b)
# #     episode_reward += b
#     new_env.render()
# #     print(episode_reward)
#     time.sleep(0.05)


#
# while True:
#    frame=[]s
#    a,b,c,d=new_env.step(6)
#    print("reward is",b)
#    # new_env.step(3)
#    # new_env.step(1)
#    # new_env.step(5)
#    # a,b,c,d=new_env.step(6)
#    # print(b)
#    new_env.render()
# #
#    time.sleep(0.05)
# action_space=new_env.action_space
#
# pdtype = make_pdtype(action_space)
#
#
# while True:
#    a0= pdtype.sample()
#
#    a,b,c,d=new_env.step(a0)
#    print("action",a0)
#    # new_env.step(3)
#    # new_env.step(1)
#    # new_env.step(5)
#    # a,b,c,d=new_env.step(6)
#    # print(b)
#    new_env.render()
#
#    time.sleep(0.05)

