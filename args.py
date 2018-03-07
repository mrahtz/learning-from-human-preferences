import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_general_args(parser)
    add_pref_interface_args(parser)
    add_reward_predictor_args(parser)
    add_a2c_args(parser)
    args = parser.parse_args()
    return args


def add_general_args(parser):
    parser.add_argument('--env')
    # Which parts to run
    parser.add_argument('--orig_rewards', action='store_true')
    parser.add_argument('--no_gather_prefs', action='store_true')
    parser.add_argument('--no_a2c', action='store_true')
    seconds_since_epoch = str(int(time.time()))
    parser.add_argument('--run_name', default=seconds_since_epoch)
    parser.add_argument('--log_dir')
    parser.add_argument('--test_mode', action='store_true')
    parser.add_argument('--debug', action='store_true')


def add_a2c_args(parser):
    parser.add_argument('--print_lr', action='store_true')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--ent_coef', type=float, default=0.01)
    parser.add_argument('--n_envs', type=int, default=1)
    parser.add_argument("--lr_zero_million_timesteps",
                        type=float, default=None,
                        help='If set, decay learning rate linearly, reaching '
                             ' zero at this many timesteps')
    parser.add_argument('--lr', type=float, default=7e-4)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--load_policy_ckpt_dir',
                        help='Load a policy checkpoint from this directory.')
    parser.add_argument('--policy_ckpt_interval', type=int, default=100,
                        help="No. updates between policy checkpoints")
    parser.add_argument('--render_episodes', action='store_true')
    parser.add_argument('--million_timesteps',
                        type=float, default=10.,
                        help='How many million timesteps to train for. '
                             '(The number of frames trained for is this '
                             'multiplied by 4 due to frameskip.)')


def add_reward_predictor_args(parser):
    parser.add_argument('--save_initial_prefs', action='store_true')
    parser.add_argument('--skip_prefs', action='store_true')
    parser.add_argument('--save_pretrain', action='store_true')
    parser.add_argument('--just_prefs', action='store_true')
    parser.add_argument('--network', default='conv')
    parser.add_argument('--save_prefs', action='store_true')
    parser.add_argument('--no_pretrain', action='store_true')
    parser.add_argument('--just_pretrain', action='store_true')
    parser.add_argument('--db_max', type=int, default=3000)
    parser.add_argument('--load_prefs_dir')
    parser.add_argument('--rp_lr', type=float, default=2e-4)
    parser.add_argument('--n_initial_epochs', type=int, default=200)
    parser.add_argument('--n_preds', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batchnorm', action='store_true')
    parser.add_argument('--n_initial_prefs', type=int, default=500,
                        help='How many preferences to collect from a random '
                             'policy before starting reward predictor '
                             'training')
    parser.add_argument('--load_reward_predictor_ckpt',
                        help='File to load reward predictor checkpoint from '
                             '(e.g. runs/foo/reward_predictor_checkpoints/'
                             'reward_predictor.ckpt-100)')
    parser.add_argument('--reward_predictor_ckpt_interval',
                        type=int, default=100,
                        help='No. training batches between reward '
                             'predictor checkpoints')


def add_pref_interface_args(parser):
    parser.add_argument('--random_queries', action='store_true')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--segs_max', type=int, default=1000)