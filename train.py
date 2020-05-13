from model import Model
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import flag
import datetime
import montezuma_revenge_env
from baselines import logger
from rnd_model import TargetModel,PredictorModel
from utils import RunningStdMean,RewardForwardFilter
from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import Pipe, Process
from torch.distributions.categorical import Categorical
import torch.nn.functional as F


class Trainer:
    def __init__(self,num_training_steps,num_env,num_game_steps,num_epoch,
                 learning_rate,discount_factor,int_discount_factor, num_action,
                 value_coef,clip_range,save_interval,log_interval,entropy_coef,lam,mini_batch_num,num_action_repeat,load_path,ext_adv_coef,int_adv_coef,num_pre_norm_steps):
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
        self.mini_batch_num=mini_batch_num
        self.num_action=num_action
        self.num_pre_norm_steps=num_pre_norm_steps
        self.int_discount_factor=int_discount_factor

        assert self.batch_size % self.mini_batch_num == 0
        self.mini_batch_size=int(self.batch_size / self.mini_batch_num)
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
        self.ext_adv_coef=ext_adv_coef
        self.int_adv_coef=int_adv_coef
        self.writer = SummaryWriter('logs/' + self.current_time + '/log')
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
        self.target_model = TargetModel(self.num_action).to(self.device)
        self.predictor_model = PredictorModel(self.num_action).to(self.device)
        self.mse_loss = nn.MSELoss()
        self.predictor_mse_loss=nn.MSELoss(reduction='none')

        self.reward_rms = RunningStdMean()
        self.obs_rms = RunningStdMean(shape=(1, 1, 84, 84))
        self.reward_filter = RewardForwardFilter(self.int_discount_factor)

    def collect_experiance_and_train(self):
        start_train_step = 0
        sample_episode_num = 0

        if flag.LOAD:
            checkpoint = torch.load(self.load_path)

            self.new_model.load_state_dict(checkpoint['new_model_state_dict'])
            self.predictor_model.load_state_dict(checkpoint['predictor_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_train_step = checkpoint['train_step']
            sample_episode_num = checkpoint['ep_num']
            self.obs_rms.mean = checkpoint['obs_mean']
            self.obs_rms.var = checkpoint['obs_var']
            self.obs_rms.count = checkpoint['obs_count']
            self.reward_rms.mean = checkpoint['rew_mean']
            self.reward_rms.var = checkpoint['rew_var']
            self.reward_rms.count = checkpoint['rew_count']
            self.reward_filter.rewems=checkpoint['rewems']
            print("loaded model weights from checkpoint")

        current_observations = []
        parents=[]
        childs = []
        envs=[]

        for i in range(self.num_env):
            parent,child = Pipe()
            new_env = montezuma_revenge_env.MontezumaRevenge(i,child,self.num_action_repeat,0.25)
            new_env.start()
            envs.append(new_env)
            parents.append(parent)
            childs.append(child)
        if flag.LOAD:

            actions=np.random.randint(0,self.num_action,size=(self.num_env))

            for i in range(0,len(parents)):
                parents[i].send(actions[i])
            current_observations=[]
            for i in range(0,len(parents)):
                obs, rew , done = parents[i].recv()
                current_observations.append(obs)
        else:
        #normalize observations

            observations_to_normalize=[]
            for step in range(self.num_game_steps * self.num_pre_norm_steps):

                actions = np.random.randint(0,self.num_action,size=(self.num_env))

                for i in range(0,len(parents)):
                    parents[i].send(actions[i])
                current_observations=[]
                for i in range(0,len(parents)):
                    obs,rew , done = parents[i].recv()
                    current_observations.append(obs)
                observations_to_normalize.extend(current_observations)
                if(len(observations_to_normalize)%(self.num_game_steps*self.num_env)==0):
                    observations_to_normalize=np.stack(observations_to_normalize)[:,3,:,:].reshape(-1,1,84,84)
                    self.obs_rms.update(observations_to_normalize)
                    observations_to_normalize=[]
            print("normalization ended")

        sample_ext_reward=0

        for train_step in range(start_train_step,self.training_steps):

            total_observations=[]
            total_int_rewards=[]
            total_ext_rewards=[]
            total_dones=[]
            total_int_values=[]
            total_ext_values=[]
            total_actions=[]


            for game_step in range(self.num_game_steps):

                total_observations.extend(current_observations)

                with torch.no_grad():
                    current_observations_tensor = torch.from_numpy(np.array(current_observations)).float().to(self.device)
                    decided_actions, predicted_ext_values, predicted_int_values = self.new_model.step(current_observations_tensor)
                    one_channel_observations=np.array(current_observations)[:,3,:,:].reshape(-1,1,84,84)
                    one_channel_observations = (
                                (one_channel_observations - self.obs_rms.mean) / np.sqrt(self.obs_rms.var)).clip(-5, 5)
                    one_channel_observations_tensor=torch.from_numpy(one_channel_observations).float().to(self.device)
                    total_int_rewards.append(self.get_intrinsic_rewards(one_channel_observations_tensor))


                total_int_values.append(predicted_int_values)
                total_ext_values.append(predicted_ext_values)
                total_actions.extend(decided_actions)

                current_observations=[]
                for i in range(0, len(parents)):
                    parents[i].send(decided_actions[i])

                step_rewards = []
                step_dones = []
                for i in range(0, len(parents)):

                    observation, reward, done =parents[i].recv()
                    current_observations.append(observation)
                    step_rewards.append(reward)
                    step_dones.append(done)
                sample_ext_reward += step_rewards[0]

                if step_dones[0]:

                    self.writer.add_scalar('ext_reward_per_episode_for_one_env',  sample_ext_reward, sample_episode_num)
                    sample_ext_reward = 0
                    sample_episode_num += 1

                total_ext_rewards.append(step_rewards)
                total_dones.append(step_dones)
            # next state value, required for computing advantages
            with torch.no_grad():
                current_observations_tensor = torch.from_numpy(np.array(current_observations)).float().to(self.device)
                decided_actions, predicted_ext_values,predicted_int_values = self.new_model.step(current_observations_tensor)

            total_int_values.append(predicted_int_values)
            total_ext_values.append(predicted_ext_values)

            # convert lists to numpy arrays
            observations_array=np.array(total_observations)
            total_one_channel_observations_array=observations_array[:,3,:,:].reshape(-1,1,84,84)
            total_one_channel_observations_array = ((total_one_channel_observations_array - self.obs_rms.mean) / np.sqrt(self.obs_rms.var)).clip(-5,5)
            ext_rewards_array = np.array(total_ext_rewards)

            dones_array = np.array(total_dones)
            ext_values_array=np.array(total_ext_values)
            int_values_array = np.array(total_int_values)
            actions_array = np.array(total_actions)
            int_rewards_array = np.stack(total_int_rewards)

            total_reward_per_env = np.array([self.reward_filter.update(reward_per_step) for reward_per_step in
                                             int_rewards_array])
            mean, std, count = np.mean(total_reward_per_env), np.std(total_reward_per_env), len(total_reward_per_env)
            self.reward_rms.update_from_mean_std(mean, std ** 2, count)

            # normalize intrinsic reward
            int_rewards_array /= np.sqrt(self.reward_rms.var)
            self.writer.add_scalar('avg_int_reward_per_train_step_for_every_env', np.sum(int_rewards_array) / self.num_env, train_step)
            self.writer.add_scalar('int_reward_for_one_env_per_step',int_rewards_array.T[0].mean(),train_step)
            # print("one channel" , one_channel_observations.shape)
            # print("total observations", observations_array.shape)
            # print("ext_rewards",ext_rewards_array.shape)
            # print("ext_values",ext_values_array.shape)
            # print("int_rewards", int_rewards_array)
            # print("int_values", int_values_array.shape)
            # print("dones",dones_array.shape)
            # print("actions",actions_array.shape)

            ext_advantages_array,ext_returns_array=self.compute_advantage(ext_rewards_array,ext_values_array,dones_array,0)
            int_advantages_array, int_returns_array = self.compute_advantage(int_rewards_array, int_values_array,
                                                                             dones_array,1)

            advantages_array = self.ext_adv_coef * ext_advantages_array + self.int_adv_coef * int_advantages_array
            self.obs_rms.update(total_one_channel_observations_array)

            if flag.DEBUG:
                print("all actions are",total_actions)


            # end=time.time()

            # print("time elapsed in game steps",end-start)
            # start=time.time()


            observations_tensor=torch.from_numpy(np.array(observations_array)).float().to(self.device)
            ext_returns_tensor=torch.from_numpy(np.array(ext_returns_array)).float().to(self.device)
            int_returns_tensor = torch.from_numpy(np.array(int_returns_array)).float().to(self.device)
            actions_tensor = torch.from_numpy(np.array(actions_array)).long().to(self.device)
            advantages_tensor=torch.from_numpy(np.array(advantages_array)).float().to(self.device)
            one_channel_observations_tensor=torch.from_numpy(total_one_channel_observations_array).float().to(self.device)

            # print(observations_tensor.shape)
            # print(ext_returns_tensor.shape)
            # print(int_returns_tensor.shape)
            # print(actions_tensor.shape)
            # print(advantages_tensor.shape)
            # print(one_channel_observations_tensor.shape)
            random_indexes = np.arange(self.batch_size)
            np.random.shuffle(random_indexes)


            with torch.no_grad():
                old_policy, _,_ = self.new_model.forward_pass(observations_tensor)
                dist_old=Categorical(F.softmax(old_policy,dim=1))
                old_log_prob = dist_old.log_prob(actions_tensor)



            loss_avg=[]
            policy_loss_avg=[]
            value_loss_avg=[]
            entropy_avg=[]
            predictor_loss_avg=[]

            for epoch in range(0,self.num_epoch):
                # print("----------------next epoch----------------")

                for n in range(0,self.mini_batch_num):
                    # print("----------------next mini batch-------------")
                    start_index=n*self.mini_batch_size
                    index_slice=random_indexes[start_index:start_index+self.mini_batch_size]
                    if flag.DEBUG:
                        print("indexed chosen are:", index_slice)

                    experience_slice=(arr[index_slice] for arr in (observations_tensor,ext_returns_tensor,int_returns_tensor,actions_tensor,
                                                                   advantages_tensor,one_channel_observations_tensor))

                    loss, policy_loss, value_loss, predictor_loss, entropy=self.train_model(*experience_slice,old_log_prob[index_slice])
                    if epoch==self.num_epoch-1:
                        loss=loss.detach().cpu().numpy()
                        policy_loss = policy_loss.detach().cpu().numpy()
                        value_loss = value_loss.detach().cpu().numpy()
                        predictor_loss = predictor_loss.detach().cpu().numpy()
                        entropy = entropy.detach().cpu().numpy()
                        loss_avg.append(loss)
                        policy_loss_avg.append(policy_loss)
                        value_loss_avg.append(value_loss)
                        entropy_avg.append(entropy)
                        predictor_loss_avg.append(predictor_loss)
            # print("----------------next training step--------------")
            # end=time.time()
            # print("epoch time",end-start)
            loss_avg_result=np.array(loss_avg).mean()
            policy_loss_avg_result=np.array(policy_loss_avg).mean()
            value_loss_avg_result=np.array(value_loss_avg).mean()
            entropy_avg_result=np.array(entropy_avg).mean()
            predictor_loss_avg_result = np.array(predictor_loss_avg).mean()
            print("training step {:03d}, Epoch {:03d}: Loss: {:.3f}, policy loss: {:.3f}, value loss: {:.3f},predictor loss: {:.3f}, entopy: {:.3f} ".format(train_step,epoch,
                                                                         loss_avg_result,
                                                                        policy_loss_avg_result,
                                                                         value_loss_avg_result,
                                                                        predictor_loss_avg_result,
                                                                         entropy_avg_result))

            if flag.TENSORBOARD_AVALAIBLE:
                        self.writer.add_scalar('loss_avg', loss_avg_result, train_step)
                        self.writer.add_scalar('policy_loss_avg', policy_loss_avg_result, train_step)
                        self.writer.add_scalar('value_loss_avg', value_loss_avg_result, train_step)
                        self.writer.add_scalar('predictor_loss_avg', predictor_loss_avg_result, train_step)
                        self.writer.add_scalar('entropy_avg', entropy_avg_result, train_step)

            else:
                if train_step % self.log_interval == 0:
                    logger.record_tabular("train_step", train_step)
                    logger.record_tabular("loss", loss_avg_result)
                    logger.record_tabular("value loss",  value_loss_avg_result)
                    logger.record_tabular("policy loss", policy_loss_avg_result)
                    logger.record_tabular("predictor loss", predictor_loss_avg_result)
                    logger.record_tabular("entropy", entropy_avg_result)
                    logger.record_tabular("rewards avg", np.average(total_ext_rewards))
                    logger.record_tabular("int reward avg",np.average(total_int_rewards))
                    logger.record_tabular("epoch",self.num_epoch)
                    logger.dump_tabular()


            if train_step % self.save_interval==0:
                train_checkpoint_dir = 'logs/' + self.current_time  + str(train_step)

                torch.save({
                    'train_step': train_step,
                    'new_model_state_dict': self.new_model.state_dict(),
                    'predictor_state_dict': self.predictor_model.state_dict(),
                    'target_state_dict':self.target_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'obs_mean': self.obs_rms.mean,
                    'obs_var': self.obs_rms.var,
                    'obs_count': self.obs_rms.count,
                    'rew_mean': self.reward_rms.mean,
                    'rew_var': self.reward_rms.var,
                    'rew_count': self.reward_rms.count,
                    'rewems': self.reward_filter.rewems,
                     'ep_num': sample_episode_num


                }, train_checkpoint_dir)

              


    def compute_advantage(self, rewards, values, dones, int_flag=0):

        if flag.DEBUG:
            print("---------computing advantage---------")
            print("rewards are",rewards)
            print("values from steps are",values)
        if int_flag==1:
            discount_factor=self.int_discount_factor
        else:
            discount_factor=self.discount_factor
        advantages = []
        last_advantage = 0
        for step in reversed(range(self.num_game_steps)):
            if int_flag==1:
                is_there_a_next_state = 1
            else:
                 is_there_a_next_state = 1.0 - dones[step]
            delta = rewards[step] + (is_there_a_next_state * discount_factor * values[step + 1]) - values[step]
            if flag.USE_GAE:
                    advantage = last_advantage = delta + discount_factor * \
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


    def train_model(self,observations_tensor,ext_returns_tensor,int_returns_tensor,actions_tensor,advantages_tensor,one_channel_observations_tensor, old_log_prob):

            #
            # if flag.USE_STANDARD_ADV:
            #     advantages_array=advantages_tensor.mean() / (advantages_tensor.std() + 1e-13)
            # # print("values from steps",values_array)


            if flag.DEBUG:
                print("input observations shape", observations_tensor.shape)
                print("ext returns shape", ext_returns_tensor.shape)
                print("int returns shape", int_returns_tensor.shape)
                print("input actions shape", actions_tensor.shape)
                print("input advantages shape", advantages_tensor.shape)
                print("one channel observations", one_channel_observations_tensor.shape)






            self.new_model.train()
            self.predictor_model.train()
            target_value = self.target_model.forward_pass(one_channel_observations_tensor)
            predictor_value = self.predictor_model.forward_pass(one_channel_observations_tensor)
            predictor_loss = self.predictor_mse_loss(predictor_value, target_value).mean(-1)
            mask = torch.rand(len(predictor_loss)).to(self.device)
            mask = (mask < 0.25).type(torch.FloatTensor).to(self.device)
            predictor_loss = (predictor_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))
            new_policy, ext_new_values, int_new_values = self.new_model.forward_pass(observations_tensor)
            ext_value_loss = self.mse_loss(ext_new_values, ext_returns_tensor)
            int_value_loss = self.mse_loss(int_new_values, int_returns_tensor)
            value_loss = ext_value_loss + int_value_loss
            new_dist= Categorical(F.softmax(new_policy,dim=1))
            new_log_prob = new_dist.log_prob(actions_tensor)

            ratio = torch.exp(new_log_prob - old_log_prob)

            clipped_policy_loss = torch.clamp(ratio, 1.0 - self.clip_range, 1 + self.clip_range) * advantages_tensor
            policy_loss = ratio * advantages_tensor

            selected_policy_loss = -torch.min(clipped_policy_loss, policy_loss).mean()
            entropy = new_dist.entropy().mean()
            loss = selected_policy_loss + (self.value_coef * value_loss) - (self.entropy_coef * entropy) + predictor_loss
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.new_model.parameters(), 0.5)
            # torch.nn.utils.clip_grad_norm_(self.predictor_model.parameters(), 0.5)
            self.optimizer.step()
            return loss, selected_policy_loss, value_loss, predictor_loss, entropy


    def get_intrinsic_rewards(self,input_observation):

        target_value = self.target_model.forward_pass(input_observation) #shape: [n,512]
        predictor_value = self.predictor_model.forward_pass(input_observation) #shape [n,512]
        intrinsic_reward=(target_value - predictor_value).pow(2).sum(1) / 2
        intrinsic_reward= intrinsic_reward.detach().cpu().numpy()
        return intrinsic_reward




































