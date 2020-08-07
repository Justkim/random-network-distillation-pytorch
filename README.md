# Random Network Distilation 

> This is an implementation of Random network distillation on Montezuma's Revenge using pytorch.   
> paper link: https://arxiv.org/abs/1810.12894   


## Setup
To run the program, first install the required packages by executing:   

```shell
$ pip3 install -r requirements.txt
```

## Play 
Run the program with pretrained model and see the agent playing:

```shell
$ python3 main.py --play --path models/pretrained_model.pth
```
  
![trained agent playing](demo/mr_playing.gif)
   




![entropy](demo/entropy.png?raw=true "Entropy ")  
x-axis: train steps, y-axis: entropy
this diagram shows how entropy decreases. the agent starts by total random movements and learns a stochastic policy after being trained.  

the pretrained model `models/pretrained_model.pth` is obtained by training with the following settings:   

| variable |  value| 
|:-----|:--------:|
| environment type | "MR" |
|   number of train steps  | 11400 |  
| normilization steps parameter  | 1000   |
| number of environments   |  64 |   
| number of epoches | 4 |
| agent steps(rollout) | 128 |
| number of mini batches | 2 |
| learning rate | 0.0001 |
| discount factor | 0.999 |
| intrinsic discount factor | 0.99 | 
| lambda(related to generilized advantage estimation algorithm) | 0.95 |
| clip(related to PPO algorithm) | 0.1 |
| value loss coefficient | 0.5 |
| entropy coefficient | 0.001 |
| the predictior's update proportion | 0.25 |
| intrinsic advantages coefficient | 1 |
| extrinsic advantages coefficient | 2 |

## Train

You can train from a model from scratch by using the following command. Note that if you don't specify the variables, They match the default value described in the table above.  The save_int varibale describes the interval of saving a model checkpoint.   

Some useful diagrams are stored in tensorboard format while training.
```shell 
python3 main.py --train --num_env 64 --train_steps 12000 --predictor_update_p 0.25 --num_pre_norm_steps 10 --game_steps 128 --num_epoch 4 --mini_batch 2 --save_int 100 
```
Train from a checkpoint:

```shell 
python3 main.py ---train --path logs/desired_checkpoint --num_env 64 --train_steps 12000 --predictor_update_p 0.25 --num_pre_norm_steps 10 --game_steps 128 --num_epoch 4 --mini_batch 2 --save_int 100 
```


