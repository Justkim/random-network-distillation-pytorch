from train import Trainer
import flag
from play import Player
import moving_dot_env
import argparse

flag.TRAIN = True
flag.PLAY = False
flag.Load = False


parser=argparse.ArgumentParser(description="train parser")
parser.add_argument("--num_env", default=1, type=int, help="This is the number of workers")
parser.add_argument("--game_steps", default=1, type=int, help="This is the number of steps in game "
                                                               "for every training step")
parser.add_argument("--num_epoch", default=1, type=int, help="This is the number of epoches")
parser.add_argument("--mini_batch", default=1, type=int, help="This is mini batch size ")
parser.add_argument("--lr", default=2e-4, type=float, help="This is optimizer learning rate")
parser.add_argument("--gamma", default=0.99, type=float, help="This is discount factor")
parser.add_argument("--lambda_gae", default=0.95, type=float, help="This is lambda in GAE")
parser.add_argument("--clip_range", default=0.1, type=float, help="This is clip range for PPO")
parser.add_argument("--value_coef", default=0.5, type=float, help="This is value coef")
parser.add_argument("--ent_coef", default=0.05, type=float, help="This is entropy coef")
parser.add_argument("--log_int", default=10, type=int, help="This is log interval")
parser.add_argument("--save_int", default=1, type=int, help="This is save interval")
parser.add_argument("--action_re", default=1, type=int, help="This is number of action repeats")
parser.add_argument("--train_steps", default=8000, type=int, help="This is number of train steps")
parser.add_argument("--play", default=False,action="store_true",  help="use this if u want the network to play the game")
parser.add_argument("--load", default=False,action="store_true", help="use this if u want to load a model to continue from")
parser.add_argument("--path", default="",type=str, help="path of model to load / either for train or test")
args=parser.parse_args()

if args.play:
    flag.TRAIN=False
    flag.PLAY=True
if args.load:
    flag.LOAD=True

if flag.TRAIN:
    new_trainer = Trainer(num_training_steps=args.train_steps, num_env=args.num_env, num_game_steps=args.game_steps, num_epoch=args.num_epoch, learning_rate=args.lr
                          , discount_factor=args.gamma, num_action=7, clip_range=args.clip_range, value_coef=args.value_coef,
                          save_interval=args.save_int,
                          log_interval=args.log_int,
                          entropy_coef=args.ent_coef, lam=args.lambda_gae, mini_batch_size=args.mini_batch, num_action_repeat=args.action_re,load_path=args.path)
    new_trainer.collect_experiance_and_train()
elif flag.PLAY:
    env = moving_dot_env.make_train_0()
    new_player=Player(env=env,load_path=args.path)
    new_player.play()

# else:
#     new_player=Player(env=env)
#     new_player.play(%cd PPO)

