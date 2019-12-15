from model import Model
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import flag
import datetime
import ray
import moving_dot_env
from baselines import logger
import time
from torch.distributions.categorical import Categorical


@ray.remote
class Simulator(object):
    def __init__(self,num_action_repeat):
        self.env = moving_dot_env.make_train_0()
        self.env.reset()
        self.num_action_repeat=num_action_repeat

    def step(self, action):
        for i in range(self.num_action_repeat):
            observations,rewards,dones,info=self.env.step(action)
            if dones:
                observations = self.reset()
        if flag.SHOW_GAME:
            self.env.render()
        return observations, rewards, dones

    def reset(self):
        return self.env.reset()


class Trainer():
    def __init__(self,num_training_steps,num_env,num_game_steps,num_epoch,
                 learning_rate,discount_factor,num_action,
                 value_coef,clip_range,save_interval,log_interval,entropy_coef,lam,mini_batch_size,num_action_repeat,load_path):
        self.training_steps=num_training_steps
        self.num_epoch=num_epoch
        self.learning_rate=learning_rate
        self.discount_factor=discount_factor
        self.num_game_steps=num_game_steps
        self.num_env=num_env
        self.batch_size=num_env*num_game_steps
        self.clip_range=clip_range
        self.value_coef=value_coef
        self.entropy_coef = entropy_coef
        self.mini_batch_size=mini_batch_size
        self.num_action=num_action

        assert self.batch_size % self.mini_batch_size == 0
        self.mini_batch_num=int(self.batch_size / self.mini_batch_size)
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = 'logs/' + self.current_time + '/log'
        logger.configure(dir=log_dir)
        self.save_interval=save_interval
        self.lam=lam
        self.log_interval=log_interval

        self.num_action_repeat=num_action_repeat
        self.clip_range = clip_range

        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.load_path=load_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.new_model = Model(self.num_action).to(self.device)
        self.optimizer = optim.Adam(self.new_model.parameters(), lr=self.learning_rate)

        logger.record_tabular("time: ", self.current_time)
        logger.record_tabular("num_env: ", self.num_env)
        logger.record_tabular("steps: ", self.num_game_steps)
        logger.record_tabular("mini batch: ", self.mini_batch_size)
        logger.record_tabular("lr: ", self.learning_rate)
        logger.record_tabular("gamma: ", self.discount_factor)
        logger.record_tabular("lambda: ", self.lam)
        logger.record_tabular("clip: ", self.clip_range)
        logger.record_tabular("v_coef: ", self.value_coef)
        logger.record_tabular("ent_coef: ", self.entropy_coef)
        logger.dump_tabular()

    def collect_experiance_and_train(self):
        start_train_step = 0

        if flag.LOAD:
            checkpoint = torch.load(self.load_path)

            self.new_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_train_step= checkpoint['train_step']
            print("loaded model weights from checkpoint")

        ray.init()
        current_observations = []
        runners = []
        returned_observations = []

        for i in range(self.num_env):
            runners.append(Simulator.remote(self.num_action_repeat))
            returned_observations.append(runners[i].reset.remote())
        for i in range(self.num_env):
            current_observations.append(ray.get(returned_observations[i]))

        for train_step in range(start_train_step,self.training_steps):

            observations=[]
            rewards=[]
            dones=[]
            values=[]
            actions=[]

            start=time.time()
            cross_entropy_loss = nn.CrossEntropyLoss()

            for game_step in range(self.num_game_steps):
                returned_objects = []
                observations.extend(current_observations)
                with torch.no_grad():
                    current_observations_tensor = torch.from_numpy(np.array(current_observations)).float().to(self.device)
                    decided_actions, predicted_values = self.new_model.step(current_observations_tensor)
                values.append(predicted_values)
                actions.extend(decided_actions)
                experiences=[]
                for i in range(self.num_env):
                        returned_objects.append(runners[i].step.remote(decided_actions[i]))
                        experiences.append(ray.get(returned_objects[i]))
                current_observations=[each[0] for each in experiences]
                current_observations=np.array(current_observations)
                rewards.append([each[1] for each in experiences])
                dones.append([each[2] for each in experiences])


            # next state value, required for computing advantages
            with torch.no_grad():
                current_observations_tensor = torch.from_numpy(np.array(current_observations)).float().to(self.device)
                decided_actions, predicted_values = self.new_model.step(current_observations_tensor)
            values.append(predicted_values)

            # convert lists to numpy arrays
            observations_array=np.array(observations)
            rewards_array = np.array(rewards)
            dones_array = np.array(dones)
            values_array=np.array(values)
            actions_array = np.array(actions)

            advantages_array,returns_array=self.compute_advantage(rewards_array,values_array,dones_array)
            # values_array=values_array.flatten()

            if flag.DEBUG:
                print("all actions are",actions)

            random_indexes=np.arange(self.batch_size)
            np.random.shuffle(random_indexes)
            end=time.time()

            # print("time elapsed in game steps",end-start)
            start=time.time()

            observations_tensor=torch.from_numpy(np.array(observations_array)).float().to(self.device)
            returns_tensor=torch.from_numpy(np.array(returns_array)).float().to(self.device)
            actions_tensor = torch.from_numpy(np.array(actions_array)).long().to(self.device)
            advantages_tensor=torch.from_numpy(np.array(advantages_array)).float().to(self.device)

            with torch.no_grad():
                old_policy, old_values = self.new_model.forward_pass(observations_tensor)
                old_negative_log_p = cross_entropy_loss(old_policy, actions_tensor)
            loss_avg=[]
            policy_loss_avg=[]
            value_loss_avg=[]
            entropy_avg=[]

            for epoch in range(0,self.num_epoch):
                # print("----------------next epoch----------------")

                for n in range(0,self.mini_batch_num):
                    # print("----------------next mini batch-------------")
                    start_index=n*self.mini_batch_size
                    index_slice=random_indexes[start_index:start_index+self.mini_batch_size]
                    if flag.DEBUG:
                        print("indexed chosen are:",index_slice)

                    experience_slice=(arr[index_slice] for arr in (observations_tensor,returns_tensor,actions_tensor,
                                                                   advantages_tensor))

                    loss, policy_loss, value_loss, entropy=self.train_model(*experience_slice,old_negative_log_p)
                    loss=loss.detach().cpu().numpy()
                    policy_loss = policy_loss.detach().cpu().numpy()
                    value_loss = value_loss.detach().cpu().numpy()
                    entropy = entropy.detach().cpu().numpy()
                    #self.old_model.set_weights(last_weights)
                    loss_avg.append(loss)
                    policy_loss_avg.append(policy_loss)
                    value_loss_avg.append(value_loss)
                    entropy_avg.append(entropy)
            # print("----------------next training step--------------")

            end=time.time()
            # print("epoch time",end-start)
            loss_avg_result=np.array(loss_avg).mean()
            policy_loss_avg_result=np.array(policy_loss_avg).mean()
            value_loss_avg_result=np.array(value_loss_avg).mean()
            entropy_avg_result=np.array(entropy_avg).mean()
            print("training step {:03d}, Epoch {:03d}: Loss: {:.3f}, policy loss: {:.3f}, value loss: {:.3f}, entopy: {:.3f} ".format(train_step,epoch,
                                                                         loss_avg_result,
                                                                        policy_loss_avg_result,
                                                                         value_loss_avg_result,
                                                                         entropy_avg_result))
            # if flag.DEBUG:
            #     print("policy", self.new_model.probs)
            if flag.TENSORBOARD_AVALAIBLE:
                    #add instructions here
                    print("not implemented")
            else:
                if train_step % self.log_interval == 0:
                    logger.record_tabular("train_step", train_step)
                    logger.record_tabular("loss", loss_avg_result)
                    logger.record_tabular("value loss",  value_loss_avg_result)
                    logger.record_tabular("policy loss", policy_loss_avg_result)
                    logger.record_tabular("entropy", entropy_avg_result)
                    logger.record_tabular("rewards avg", np.average(rewards))
                    logger.dump_tabular()


            if train_step % self.save_interval==0:
                train_checkpoint_dir = 'logs/' + self.current_time + "/" + str(train_step)

                torch.save({
                    'train_step': train_step,
                    'model_state_dict': self.new_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),


                }, train_checkpoint_dir)


    def compute_advantage(self, rewards, values, dones):
        if flag.DEBUG:
            print("---------computing advantage---------")
            print("rewards are",rewards)
            print("values from steps are",values)

        advantages = []
        last_advantage = 0
        for step in reversed(range(self.num_game_steps)):
            is_there_a_next_state = 1.0 - dones[step]
            delta = rewards[step] + (is_there_a_next_state * self.discount_factor * values[step + 1]) - values[step]
            if flag.USE_GAE:
                    advantage = last_advantage = delta + self.discount_factor * \
                                                 self.lam * is_there_a_next_state * last_advantage
                    advantages.append(advantage)
            else:
                    advantages.append(delta)
        advantages.reverse()

        advantages=np.array(advantages)
        advantages = advantages.flatten()
        values=values[:-1]
        returns=advantages+values.flatten()
        if flag.DEBUG:
            print("all advantages are",advantages)
            print("all returns are",returns)
        return advantages,returns


    def train_model(self,observations_tensor,returns_tensor,actions_tensor,advantages_tensor,old_negative_log_p):

            if flag.USE_STANDARD_ADV:
                advantages_array=advantages_tensor.mean() / (advantages_tensor.std() + 1e-13)
            # print("values from steps",values_array)

            if flag.DEBUG:
                print("input observations shape", observations_tensor.shape)
                print("input rewards shape", returns_tensor.shape)
                print("input actions shape", actions_tensor.shape)
                print("input advantages shape", advantages_tensor.shape)

                print("returns",returns_tensor)
                print("advantages",advantages_tensor)
                print("actions",actions_tensor)


            loss,policy_loss,value_loss,entropy=self.do_train(observations_tensor,returns_tensor,actions_tensor, advantages_tensor,old_negative_log_p)
            return loss,policy_loss,value_loss,entropy

    def do_train(self,observations,returns,actions, advantages, old_negative_log_p):
        cross_entropy_loss = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()

        self.new_model.train()
        new_policy, new_values= self.new_model.forward_pass(observations)

        value_loss=mse_loss(new_values,returns)
        new_negative_log_p = cross_entropy_loss(new_policy,actions)
        ratio= torch.exp(old_negative_log_p - new_negative_log_p)

        clipped_policy_loss=torch.clamp(ratio,1.0-self.clip_range, 1+self.clip_range)*advantages
        policy_loss=ratio*advantages

        selected_policy_loss=-torch.min(clipped_policy_loss,policy_loss).mean()
        dist = Categorical(logits=new_policy)
        entropy=dist.entropy().mean()
        loss = selected_policy_loss + self.value_coef*value_loss - self.entropy_coef * entropy
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.new_model.parameters(),0.5)
        self.optimizer.step()
        return loss, policy_loss, value_loss, entropy



































