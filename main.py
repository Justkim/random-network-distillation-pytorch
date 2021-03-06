from train import Trainer
import flag
from play import Player
import argparse


parser = argparse.ArgumentParser(description="train parser")
parser.add_argument("--num_env", default=64, type=int,
                    help="This is the number of workers")
parser.add_argument("--game_steps", default=128, type=int,
                    help="This is the number of steps in one environment "
                         "for every training step")
parser.add_argument("--num_epoch", default=4, type=int,
                    help="This is the number of epoches")
parser.add_argument("--mini_batch", default=2, type=int,
                    help="This is mini batch size ")
parser.add_argument("--lr", default=1e-4, type=float,
                    help="This is optimizer learning rate")
parser.add_argument("--gamma", default=0.999, type=float,
                    help="This is discount factor")
parser.add_argument("--int_gamma", default=0.99, type=float,
                    help="This is the intrinsic discount factor")
parser.add_argument("--lambda_gae", default=0.95, type=float,
                    help="This is lambda in GAE")
parser.add_argument("--clip_range", default=0.1, type=float,
                    help="This is clip range for PPO")
parser.add_argument("--value_coef", default=0.5, type=float,
                    help="This is value coef")
parser.add_argument("--ent_coef", default=0.001, type=float,
                    help="This is entropy coef")
parser.add_argument("--save_int", default=500, type=int,
                    help="This is save interval")
parser.add_argument("--action_re", default=4, type=int,
                    help="This is number of action repeats")
parser.add_argument("--train_steps", default=3000, type=int,
                    help="This is number of train steps")
parser.add_argument("--play", default=False, action="store_true",
                    help="use this if u want the network to play the game")
parser.add_argument("--train", default=False, action="store_true",
                    help="use this for training the agent")
parser.add_argument("--path", default="", type=str,
                    help="path of model to load / either for train or test")
parser.add_argument("--ext_adv_coef", default=2, type=float,
                    help="extrinsic advantage coef")
parser.add_argument("--int_adv_coef", default=1, type=float,
                    help="intrinsic advantage coef")
parser.add_argument("--env_type", default="MR", type=str)
parser.add_argument("--predictor_update_p", default=1, type=float,
                    help="the amount of predictor update by probability")
parser.add_argument("--num_pre_norm_steps", default=1000, type=int,
                    help="the number of steps taken before game"
                    "for initializing normilization"
                    " is: this variable * game_steps"
                    )


args = parser.parse_args()

if args.play:
    flag.TRAIN = False
    flag.PLAY = True
elif args.train:
    flag.TRAIN = True
    flag.PLAY = False
else:
    print("Please specify program running mode: play/train")
    quit()

if args.play and args.path == "":
    print("please specify the path that the model should be laoded from.")
    quit()

if args.path != "":
    flag.LOAD = True
else:
    flag.LOAD = False


if args.env_type == "MR":
    num_action = 18  # number of actions for montezuma revenge
    flag.ENV = "MR"

if flag.TRAIN:
    new_trainer = Trainer(num_training_steps=args.train_steps,
                          num_env=args.num_env, num_game_steps=args.game_steps,
                          num_epoch=args.num_epoch, learning_rate=args.lr,
                          discount_factor=args.gamma,
                          int_discount_factor=args.int_gamma,
                          num_action=num_action, clip_range=args.clip_range,
                          value_coef=args.value_coef,
                          save_interval=args.save_int,
                          entropy_coef=args.ent_coef, lam=args.lambda_gae,
                          mini_batch_num=args.mini_batch,
                          num_action_repeat=args.action_re,
                          load_path=args.path, ext_adv_coef=args.ext_adv_coef,
                          int_adv_coef=args.int_adv_coef,
                          num_pre_norm_steps=args.num_pre_norm_steps,
                          predictor_update_proportion=args.predictor_update_p)
    new_trainer.collect_experiance_and_train()
elif flag.PLAY:

    new_player = Player(load_path=args.path)
    new_player.play()
