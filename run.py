#!/bin/sh
''''exec python -u -- "$0" ${1+"$@"} # '''
import logging
import os
import os.path as osp
import sys
import subprocess
from multiprocessing import Process, Queue
import pref_interface

import easy_tf_log
import gym
import gym_moving_dot

import params
from args import parse_args
from enduro_wrapper import EnduroWrapper
from openai_baselines import logger
from openai_baselines.a2c.a2c import learn
from openai_baselines.a2c.policies import MlpPolicy, CnnPolicy
from openai_baselines.common import set_global_seeds
from openai_baselines.common.atari_wrappers import wrap_deepmind_nomax
from openai_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from pref_interface import PrefInterface
from reward_predictor import RewardPredictorEnsemble, train_reward_predictor
from utils import vid_proc, get_port_range

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # filter out INFO messages


def configure_a2c_logger(log_dir):
    a2c_dir = osp.join(log_dir, 'a2c')
    os.makedirs(a2c_dir)
    tb = logger.TensorBoardOutputFormat(a2c_dir)
    logger.Logger.CURRENT = logger.Logger(dir=a2c_dir, output_formats=[tb])


def run_parts(parts_to_run, log_dir, start_policy_training_flag=None, seg_pipe=None, episode_vid_queue=None, pref_pipe=None,
              a2c_args=None, pref_interface_args=None, train_reward_predictor_args=None):

    if 'train_reward' in parts_to_run:
        train_proc = start_reward_predictor_training(cluster_dict=cluster_dict, log_dir=log_dir,
                                                     pref_pipe=pref_pipe, go_pipe=go_pipe,
                                                     **train_reward_predictor_args)

    if 'train_reward' not in parts_to_run:
        start_policy_training_flag.put(True)

def run(general_args, a2c_args, pref_interface_args, reward_predictor_training_args):

    seg_pipe = Queue()
    pref_pipe = Queue()
    start_policy_training_flag = Queue(maxsize=1)

    """
    reward_predictor = RewardPredictorEnsemble(
        name='train_reward',
        cluster_dict=cluster_dict,
        log_dir=log_dir,
        ckpt_path=rp_ckpt_path,
        lr=rp_lr)
    """

    def reward_predictor_thunk(name, cluster_dict):
        reward_predictor = RewardPredictorEnsemble(
            name=name,
            cluster_dict=cluster_dict,
            batchnorm=reward_predictor_training_args['batchnorm'],
            dropout=reward_predictor_training_args['dropout'])
        return reward_predictor

    if general_args['render_episodes']:
        episode_vid_queue = start_episode_renderer()
    else:
        episode_vid_queue = None

    if general_args['mode'] == 'gather_initial_prefs':
        env = make_envs(general_args['env_id'], a2c_args['n_envs'], a2c_args['seed'])
        a2c_proc = start_policy_training(cluster_dict={}, log_dir=general_args['log_dir'],
                                         go_pipe=start_policy_training_flag, seg_pipe=seg_pipe, episode_vid_queue=episode_vid_queue,
                                         reward_predictor=None,
                                         gen_segments=True,
                                         env=env,
                                         **a2c_args)
        pi_proc = start_pref_interface(seg_pipe=seg_pipe, pref_pipe=pref_pipe,
                                       **pref_interface_args)
        pref_db_train, pref_db_val = pref_interface.get_initial_prefs(pref_pipe=pref_pipe,
                                                                      n_initial_prefs=general_args['n_initial_prefs'],
                                                                      db_max=general_args['db_max'])
        pref_interface.save_prefs(pref_db_train=pref_db_train, pref_db_val=pref_db_val,
                                  save_dir=general_args['log_dir'], name='initial')
        pi_proc.terminate()
        a2c_proc.terminate()
        env.close()
        return
    elif general_args['mode'] == 'pretrain_reward_predictor':
        # load preferences
        # start reward trainer
        # wait until finished
        # save checkpoint
        parts = ['train_reward_predictor']
    elif general_args['mode'] == 'train_policy_with_original_rewards':
        # just start a2c
        parts = ['a2c']
    elif general_args['mode'] == 'train_policy_with_preferences':
        # if load preferences: load preferences; else wait for initial 500
        # start a2c
        # start pref interface
        # start start reward predictor
        parts = ['a2c', 'pref_interface', 'train_reward_predictor']
        if load_prefs_dir:
            print("Loading preferences...")
            pref_db_train, pref_db_val = load_pref_db(load_prefs_dir)
        else:
            pref_db_val = PrefDB()
            pref_db_train = PrefDB()
        print("Finished accumulating initial preferences at",
              str(datetime.datetime.now()))
    else:
        raise Exception("Unknown mode: {}".format(general_args['mode']))

    # TODO
    #env.close()  # Kill the SubprocVecEnv processes
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


def start_pref_interface(seg_pipe, pref_pipe, segs_max, headless):
    def pi_procf(pi):
        # TODO hack
        sys.stdin = os.fdopen(0)
        pi.run(seg_pipe, pref_pipe, segs_max)
    pi = PrefInterface(headless=headless, synthetic_prefs=headless)
    pi_proc = Process(target=pi_procf, daemon=True, args=(pi, ))
    pi_proc.start()
    return pi_proc


def start_policy_training(cluster_dict, env, seed, seg_pipe, go_pipe, gen_segments,
                          log_dir, lr_scheduler, num_timesteps, ckpt_dir, episode_vid_queue, ent_coef, ckpt_interval,
                          n_envs, reward_predictor):
    if env.env_id == 'MovingDotNoFrameskip-v0':
        policy_fn = MlpPolicy
    elif env.env_id == 'PongNoFrameskip-v4' or env.env_id == 'EnduroNoFrameskip-v4':
        policy_fn = CnnPolicy
    else:
        raise Exception("Unsure about policy network architecture for {}".format(env.env_id))
    configure_a2c_logger(log_dir)

    def a2c():
        learn(
            policy=policy_fn,
            env=env,
            seed=seed,
            seg_pipe=seg_pipe,
            go_pipe=go_pipe,
            log_dir=log_dir,
            lr_scheduler=lr_scheduler,
            total_timesteps=num_timesteps,
            load_path=ckpt_dir,
            reward_predictor=reward_predictor,
            episode_vid_queue=episode_vid_queue,
            ent_coef=ent_coef,
            save_interval=ckpt_interval,
            gen_segments=gen_segments)
    a2c_proc = Process(target=a2c, daemon=True)
    a2c_proc.start()
    return a2c_proc


def start_reward_predictor_training(cluster_dict, log_dir, rp_ckpt_path, pref_pipe, go_pipe, load_prefs_dir, db_max, rp_lr):
    def trp():
        reward_predictor = None  # TODO XX
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
    return episode_vid_queue


def make_envs(env_id, n_envs, seed):
    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            # TODO
            if env_id == 'EnduroNoFrameskip-v4':
                env = EnduroWrapper(env)
            gym.logger.setLevel(logging.WARN)
            return wrap_deepmind_nomax(env)
        return _thunk

    set_global_seeds(seed)
    env = SubprocVecEnv(env_id, [make_env(i) for i in range(n_envs)])

    return env


def main():
    general_args, a2c_args, pref_interface_args, reward_predictor_training_args = parse_args()

    misc_logs_dir = osp.join(general_args['log_dir'], 'misc')
    os.makedirs(misc_logs_dir)
    easy_tf_log.set_dir(misc_logs_dir)

    run(general_args=general_args, a2c_args=a2c_args,
        pref_interface_args=pref_interface_args, reward_predictor_training_args=reward_predictor_training_args)

if __name__ == '__main__':
    main()
