import argparse
import time
from openai_baselines.a2c.utils import Scheduler
import os.path as osp
import sys
import subprocess
import os


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_general_args(parser)
    add_pref_interface_args(parser)
    add_reward_predictor_args(parser)
    add_a2c_args(parser)
    args = parser.parse_args()

    log_dir = get_log_dir(args)
    if args.mode == 'pretrain_reward_predictor' and args.load_prefs_dir is None:
        raise Exception("Error: please specify preference databases to train with (--load_prefs_dir)")
    general_args = {
        'env_id': args.env,
        'mode': args.mode,
        'run_name': args.run_name,
        'test_mode': args.test_mode,
        'render_episodes': args.render_episodes,
        'n_initial_prefs': args.n_initial_prefs,
        'db_max': args.db_max,
        'log_dir': log_dir,
        'prefs_dir': args.load_prefs_dir
    }

    num_timesteps = int(args.million_timesteps * 1e6)
    if args.lr_zero_million_timesteps is None:
        schedule = 'constant'
        nvalues = 1  # ignored
    else:
        schedule = 'linear'
        nvalues = int(args.lr_zero_million_timesteps * 1e6)
    lr_scheduler = Scheduler(v=args.lr, nvalues=nvalues, schedule=schedule)
    a2c_args = {
        'ent_coef': args.ent_coef,
        'n_envs': args.n_envs,
        'seed': args.seed,
        'ckpt_dir': args.load_policy_ckpt_dir,
        'ckpt_interval': args.policy_ckpt_interval,
        'num_timesteps': num_timesteps,
        'lr_scheduler': lr_scheduler
    }

    pref_interface_args = {
        'headless': args.headless,
        'segs_max': args.segs_max
    }

    reward_predictor_training_args = {
        'network': args.network,
        'n_initial_epochs': args.n_initial_epochs,
        'dropout': args.dropout,
        'batchnorm': args.batchnorm,
        'ckpt_path': args.load_reward_predictor_ckpt,
        'ckpt_interval': args.reward_predictor_ckpt_interval,
        'lr': args.reward_predictor_learning_rate
    }

    if general_args['test_mode']:
        reward_predictor_training_args['val_interval'] = 1
        # Override specified arguments
        general_args['n_initial_prefs'] = 1
        reward_predictor_training_args['n_initial_epochs'] = 2
        reward_predictor_training_args['ckpt_interval'] = 1
        a2c_args['ckpt_interval'] = 10
        a2c_args['num_timesteps'] = 500
    else:
        reward_predictor_training_args['val_interval'] = 50

    with open(osp.join(log_dir, 'args.txt'), 'w') as args_file:
        args_file.write(' '.join(sys.argv))
        args_file.write('\n')
        args_file.write(str(args))

    return general_args, a2c_args, pref_interface_args, reward_predictor_training_args


def get_log_dir(args):
    if args.log_dir is not None:
        log_dir = args.log_dir
    else:
        git_rev = get_git_rev()
        run_name = args.run_name + '_' + git_rev
        log_dir = osp.join('runs', run_name)
        if osp.exists(log_dir):
            raise Exception("Log directory '%s' already exists" % log_dir)
        os.makedirs(log_dir)
    return log_dir


def get_git_rev():
    if not osp.exists('.git'):
        git_rev = "unkrev"
    else:
        git_rev = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().rstrip()
    return git_rev


def add_general_args(parser):
    parser.add_argument('mode', choices=['gather_initial_prefs', 'pretrain_reward_predictor',
                                         'train_policy_with_preferences', 'train_policy_with_original_rewards'])
    parser.add_argument('env')

    parser.add_argument('--test_mode', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--render_episodes', action='store_true')
    parser.add_argument('--load_prefs_dir')
    parser.add_argument('--n_initial_prefs', type=int, default=500,
                        help='How many preferences to collect from a random '
                             'policy before starting reward predictor '
                             'training')
    parser.add_argument('--db_max', type=int, default=3000)

    group = parser.add_mutually_exclusive_group();
    group.add_argument('--log_dir')
    seconds_since_epoch = str(int(time.time()))
    group.add_argument('--run_name', default=seconds_since_epoch)


def add_a2c_args(parser):
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--ent_coef', type=float, default=0.01)
    parser.add_argument('--n_envs', type=int, default=1)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)

    parser.add_argument("--lr_zero_million_timesteps",
                        type=float, default=None,
                        help='If set, decay learning rate linearly, reaching '
                             ' zero at this many timesteps')
    parser.add_argument('--lr', type=float, default=7e-4)

    parser.add_argument('--load_policy_ckpt_dir',
                        help='Load a policy checkpoint from this directory.')
    parser.add_argument('--policy_ckpt_interval', type=int, default=100,
                        help="No. updates between policy checkpoints")

    parser.add_argument('--million_timesteps',
                        type=float, default=10.,
                        help='How many million timesteps to train for. '
                             '(The number of frames trained for is this '
                             'multiplied by 4 due to frameskip.)')


def add_reward_predictor_args(parser):
    parser.add_argument('--network', choices=['moving_dot_features', 'cnn'], default='cnn')
    parser.add_argument('--reward_predictor_learning_rate', type=float, default=2e-4)
    parser.add_argument('--n_initial_epochs', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batchnorm', action='store_true')
    parser.add_argument('--load_reward_predictor_ckpt',
                        help='File to load reward predictor checkpoint from '
                             '(e.g. runs/foo/reward_predictor_checkpoints/'
                             'reward_predictor.ckpt-100)')
    parser.add_argument('--reward_predictor_ckpt_interval',
                        type=int, default=1,
                        help='No. training epochs between reward '
                             'predictor checkpoints')


def add_pref_interface_args(parser):
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--segs_max', type=int, default=1000)
