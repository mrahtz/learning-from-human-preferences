#!/bin/sh
''''exec python -u -- "$0" ${1+"$@"} # '''
import logging
import os
import os.path as osp
import sys
import subprocess
from multiprocessing import Process, Queue

import easy_tf_log
import gym
import gym_moving_dot

import params
from args import parse_args
from enduro_wrapper import EnduroWrapper
from openai_baselines import logger
from openai_baselines.a2c.a2c import learn
from openai_baselines.a2c.policies import MlpPolicy, CnnPolicy
from openai_baselines.a2c.utils import Scheduler
from openai_baselines.common import set_global_seeds
from openai_baselines.common.atari_wrappers import wrap_deepmind_nomax
from openai_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from pref_interface import PrefInterface
from reward_predictor import RewardPredictorEnsemble, train_reward_predictor
from utils import vid_proc, get_port_range, profile_memory

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # filter out INFO messages


def configure_a2c_logger(log_dir):
    a2c_dir = osp.join(log_dir, 'a2c')
    os.makedirs(a2c_dir)
    tb = logger.TensorBoardOutputFormat(a2c_dir)
    logger.Logger.CURRENT = logger.Logger(dir=a2c_dir, output_formats=[tb])


def run(env_id, num_timesteps, seed, lr_scheduler, rp_lr, n_envs,
        rp_ckpt_path, load_prefs_dir, headless, log_dir, ent_coef, db_max,
        segs_max, log_interval, policy_ckpt_dir, policy_ckpt_interval):
    configure_a2c_logger(log_dir)

    env = make_envs(env_id, n_envs, seed)

    seg_pipe = Queue(maxsize=100)
    pref_pipe = Queue(maxsize=100)
    go_pipe = Queue(maxsize=1)

    parts_to_run = ['a2c', 'pref_interface', 'train_reward']
    if params.params['no_gather_prefs'] or params.params['orig_rewards']:
        parts_to_run.remove('pref_interface')
    if params.params['orig_rewards']:
        parts_to_run.remove('train_reward')
    if params.params['no_a2c']:
        parts_to_run.remove('a2c')
    cluster_dict = create_cluster_dict(parts_to_run)

    if params.params['render_episodes']:
        start_episode_renderer()
    else:
        episode_vid_queue = None

    ps_proc = start_parameter_server(cluster_dict)

    if 'a2c' in parts_to_run:
        a2c_proc = start_policy_training(cluster_dict=cluster_dict, env=env, seed=seed, seg_pipe=seg_pipe,
                                         go_pipe=go_pipe, log_dir=log_dir, lr_scheduler=lr_scheduler, num_timesteps=num_timesteps,
                                         policy_ckpt_dir=policy_ckpt_dir, episode_vid_queue=episode_vid_queue,
                                         ent_coef=ent_coef, policy_ckpt_interval=policy_ckpt_interval, log_interval=log_interval)

        m1 = profile_memory(log_dir + '/mem_a2c.log', a2c_proc.pid)

    if 'pref_interface' in parts_to_run:
        pi_proc = start_pref_interface(cluster_dict=cluster_dict, seg_pipe=seg_pipe, pref_pipe=pref_pipe,
                                       segs_max=segs_max, headless=headless)

        m3 = profile_memory(log_dir + '/mem_pi.log', pi_proc.pid)
    if 'train_reward' in parts_to_run:
        train_proc = start_reward_predictor_training(cluster_dict=cluster_dict,
                                                     log_dir=log_dir,
                                                     rp_ckpt_path=rp_ckpt_path,
                                                     pref_pipe=pref_pipe,
                                                     go_pipe=go_pipe,
                                                     load_prefs_dir=load_prefs_dir,
                                                     db_max=db_max,
                                                     rp_lr=rp_lr)
        m2 = profile_memory(log_dir + '/mem_trp.log', train_proc.pid)

    if 'train_reward' not in parts_to_run:
        go_pipe.put(True)

    if params.params['just_prefs'] or params.params['just_pretrain']:
        train_proc.join()
    elif a2c_proc:
        a2c_proc.join()
    else:
        raise Exception("Error: no parts to wait for?")

    env.close()  # Kill the SubprocVecEnv processes
    if 'train_proc' in parts_to_run:
        train_proc.terminate()
    if 'pref_interface' in parts_to_run:
        pi_proc.terminate()
    ps_proc.terminate()


def create_cluster_dict(parts_to_run):
    ports = get_port_range(
        start_port=2200,
        n_ports=len(parts_to_run) + 1,
        random_stagger=True)
    cluster_dict = {'ps': ['localhost:{}'.format(ports[0])]}
    for part, port in zip(parts_to_run, ports[1:]):
        cluster_dict[part] = ['localhost:{}'.format(port)]
    return cluster_dict


def start_pref_interface(cluster_dict, seg_pipe, pref_pipe, segs_max, headless):
    def pi_procf(pi):
        # We have to give PrefInterface the reward predictor /after/ init,
        # because init needs to spawn a GUI process (using fork()), and the
        # Objective C APIs (invoked when dealing with GUI stuff) aren't
        # happy if running in a process forked from a multithreaded parent.
        reward_predictor = RewardPredictorEnsemble(
            name='pref_interface',
            cluster_dict=cluster_dict)
        pi.init_reward_predictor(reward_predictor)
        pi.run(seg_pipe, pref_pipe, segs_max)
    pi = PrefInterface(headless=headless, synthetic_prefs=headless)
    pi_proc = Process(target=pi_procf, daemon=True, args=(pi,))
    pi_proc.start()
    return pi_proc


def start_policy_training(cluster_dict, env, seed, seg_pipe, go_pipe,
                          log_dir, lr_scheduler, num_timesteps, policy_ckpt_dir, episode_vid_queue, ent_coef, policy_ckpt_interval,
                          log_interval):
    if params.params['policy'] == 'mlp':
        policy_fn = MlpPolicy
    elif params.params['policy'] == 'cnn':
        policy_fn = CnnPolicy
    else:
        raise Exception("Unknown policy {}".format(params.params['policy']))
    def a2c():
        if params.params['orig_rewards']:
            reward_predictor = None
        else:
            reward_predictor = RewardPredictorEnsemble(
                name='a2c',
                cluster_dict=cluster_dict)
        learn(
            policy=policy_fn,
            env=env,
            seed=seed,
            seg_pipe=seg_pipe,
            go_pipe=go_pipe,
            log_dir=log_dir,
            lr_scheduler=lr_scheduler,
            total_timesteps=num_timesteps,
            log_interval=log_interval,
            load_path=policy_ckpt_dir,
            reward_predictor=reward_predictor,
            episode_vid_queue=episode_vid_queue,
            ent_coef=ent_coef,
            save_interval=policy_ckpt_interval)
    a2c_proc = Process(target=a2c, daemon=True)
    a2c_proc.start()
    return a2c_proc


def start_reward_predictor_training(cluster_dict, log_dir, rp_ckpt_path, pref_pipe, go_pipe, load_prefs_dir, db_max, rp_lr):
    def trp():
        reward_predictor = RewardPredictorEnsemble(
            name='train_reward',
            cluster_dict=cluster_dict,
            log_dir=log_dir,
            ckpt_path=rp_ckpt_path,
            lr=rp_lr)
        train_reward_predictor(
            reward_predictor=reward_predictor,
            pref_pipe=pref_pipe,
            go_pipe=go_pipe,
            load_prefs_dir=load_prefs_dir,
            log_dir=log_dir,
            db_max=db_max)
    train_proc = Process(target=trp, daemon=True)
    train_proc.start()
    return train_proc


def start_parameter_server(cluster_dict):
    def ps():
        RewardPredictorEnsemble(name='ps', cluster_dict=cluster_dict)

    ps_proc = Process(target=ps, daemon=True)
    ps_proc.start()
    return ps_proc


def start_episode_renderer():
    def episode_vid_proc():
        vid_proc(
            episode_vid_queue,
            playback_speed=2,
            zoom_factor=2,
            mode='play_through')
    episode_vid_queue = Queue()
    Process(
        target=episode_vid_proc,
        daemon=True).start()


def make_envs(env_id, n_envs, seed):
    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            if params.params['env'] == 'EnduroNoFrameskip-v4':
                env = EnduroWrapper(env)
            gym.logger.setLevel(logging.WARN)
            return wrap_deepmind_nomax(env)
        return _thunk

    set_global_seeds(seed)
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    return env


def main():
    args = parse_args()

    params.init_params()
    params.params['just_pretrain'] = args.just_pretrain
    params.params['debug'] = args.debug
    params.params['no_pretrain'] = args.no_pretrain
    params.params['save_prefs'] = args.save_prefs
    params.params['network'] = args.network
    params.params['just_prefs'] = args.just_prefs
    params.params['save_pretrain'] = args.save_pretrain
    params.params['print_lr'] = args.print_lr
    params.params['env'] = args.env
    params.params['orig_rewards'] = args.orig_rewards
    params.params['skip_prefs'] = args.skip_prefs
    params.params['batchnorm'] = args.batchnorm
    params.params['dropout'] = args.dropout
    params.params['n_preds'] = args.n_preds
    params.params['save_initial_prefs'] = args.save_initial_prefs
    params.params['random_query'] = args.random_queries
    params.params['no_gather_prefs'] = args.no_gather_prefs
    params.params['no_a2c'] = args.no_a2c
    params.params['render_episodes'] = args.render_episodes

    if args.test_mode:
        print("=== WARNING: running in test mode", file=sys.stderr)
        params.params['n_initial_prefs'] = 2
        params.params['n_initial_epochs'] = 1
        params.params['prefs_save_interval'] = 1
        params.params['reward_predictor_ckpt_interval'] = 1
        params.params['reward_predictor_val_interval'] = 1
        num_timesteps = 1000
    else:
        params.params['n_initial_prefs'] = args.n_initial_prefs
        params.params['n_initial_epochs'] = args.n_initial_epochs
        params.params['prefs_save_interval'] = 10
        params.params['reward_predictor_ckpt_interval'] = \
            args.reward_predictor_ckpt_interval
        params.params['reward_predictor_val_interval'] = 50
        num_timesteps = int(args.million_timesteps * 1e6)

    if args.env == 'MovingDotNoFrameskip-v0':
        params.params['policy'] = 'mlp'
    elif args.env == 'PongNoFrameskip-v4':
        params.params['policy'] = 'cnn'
    elif args.env == 'EnduroNoFrameskip-v4':
        params.params['policy'] = 'cnn'
    else:
        raise Exception("Policy unknown for env {}".format(args.env))

    if not osp.exists('.git'):
        git_rev = "unkrev"
    else:
        git_rev = subprocess.check_output(['git', 'rev-parse', '--short',
                                           'HEAD']).rstrip().decode()
    run_name = args.run_name + '_' + git_rev
    if args.log_dir is not None:
        log_dir = args.log_dir
    else:
        log_dir = osp.join('runs', run_name)
        if osp.exists(log_dir):
            raise Exception("Log directory '%s' already exists" % log_dir)
        os.makedirs(log_dir)

    with open(osp.join(log_dir, 'args.txt'), 'w') as args_file:
        print(args, file=args_file)
        print(params.params, file=args_file)

    if args.lr_zero_million_timesteps is None:
        schedule = 'constant'
        nvalues = 1  # ignored
    else:
        schedule = 'linear'
        nvalues = int(args.lr_zero_million_timesteps * 1e6)
    lr_scheduler = Scheduler(v=args.lr,
                             nvalues=nvalues,
                             schedule=schedule)

    misc_logs_dir = osp.join(log_dir, 'misc')
    os.makedirs(misc_logs_dir)
    easy_tf_log.set_dir(misc_logs_dir)

    run(
        env_id=args.env,
        num_timesteps=num_timesteps,
        seed=args.seed,
        lr_scheduler=lr_scheduler,
        rp_lr=args.rp_lr,
        n_envs=args.n_envs,
        rp_ckpt_path=args.load_reward_predictor_ckpt,
        load_prefs_dir=args.load_prefs_dir,
        headless=args.headless,
        log_dir=log_dir,
        ent_coef=args.ent_coef,
        db_max=args.db_max,
        segs_max=args.segs_max,
        log_interval=args.log_interval,
        policy_ckpt_dir=args.load_policy_ckpt_dir,
        policy_ckpt_interval=args.policy_ckpt_interval)


if __name__ == '__main__':
    main()
