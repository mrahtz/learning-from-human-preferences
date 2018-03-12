#!/usr/bin/env python3

import logging
import os
import sys
import time
from multiprocessing import Process, Queue
from os import path as osp

import easy_tf_log
import gym
import gym_moving_dot

from params import parse_args
from enduro_wrapper import EnduroWrapper
from openai_baselines import logger
from openai_baselines.a2c.a2c import learn
from openai_baselines.a2c.policies import MlpPolicy, CnnPolicy
from openai_baselines.common import set_global_seeds
from openai_baselines.common.atari_wrappers import wrap_deepmind_nomax
from openai_baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from pref_interface import PrefInterface
from reward_predictor import RewardPredictorEnsemble
from utils import vid_proc, get_port_range
from pref_db import PrefDB, load_pref_db, save_prefs, recv_prefs

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # filter out INFO messages


def main():
    general_params, a2c_params, pref_interface_params, rew_pred_training_params = parse_args()

    run(general_params,
        a2c_params,
        pref_interface_params,
        rew_pred_training_params)


def run(general_params, a2c_params, pref_interface_params, rew_pred_training_params):
    seg_pipe = Queue()
    pref_pipe = Queue()
    start_policy_training_flag = Queue(maxsize=1)

    if general_params['render_episodes']:
        episode_vid_queue, episode_renderer_proc = start_episode_renderer()
    else:
        episode_vid_queue = episode_renderer_proc = None

    def make_reward_predictor(name, cluster_dict):
        return RewardPredictorEnsemble(
            name=name,
            cluster_dict=cluster_dict,
            log_dir=general_params['log_dir'],
            batchnorm=rew_pred_training_params['batchnorm'],
            dropout=rew_pred_training_params['dropout'],
            lr=rew_pred_training_params['lr'],
            network=rew_pred_training_params['network'])

    if general_params['mode'] == 'gather_initial_prefs':
        env, a2c_proc = start_policy_training(cluster_dict=None,
                                              make_reward_predictor=None,
                                              gen_segments=True,
                                              start_policy_training_pipe=start_policy_training_flag,
                                              seg_pipe=seg_pipe,
                                              episode_vid_queue=episode_vid_queue,
                                              log_dir=general_params['log_dir'],
                                              a2c_params=a2c_params)
        pi_proc = start_pref_interface(seg_pipe=seg_pipe, pref_pipe=pref_pipe, log_dir=general_params['log_dir'],
                                       **pref_interface_params)
        pref_db_train, pref_db_val = get_initial_prefs(pref_pipe=pref_pipe,
                                                       n_initial_prefs=general_params['n_initial_prefs'],
                                                       max_prefs=general_params['max_prefs'])
        save_prefs(pref_db_train=pref_db_train, pref_db_val=pref_db_val,
                   save_dir=general_params['log_dir'], name='initial')
        pi_proc.terminate()
        a2c_proc.terminate()
        env.close()
    elif general_params['mode'] == 'pretrain_reward_predictor':
        cluster_dict = create_cluster_dict(['ps', 'train'])
        ps_proc = start_parameter_server(cluster_dict, make_reward_predictor)
        rpt_proc = start_rew_pred_training(cluster_dict=cluster_dict,
                                           make_reward_predictor=make_reward_predictor,
                                           just_pretrain=True,
                                           pref_pipe=pref_pipe,
                                           start_policy_training_pipe=start_policy_training_flag,
                                           max_prefs=general_params['max_prefs'],
                                           prefs_dir=general_params['prefs_dir'],
                                           ckpt_path=None,
                                           n_initial_prefs=general_params['n_initial_prefs'],
                                           n_initial_epochs=rew_pred_training_params['n_initial_epochs'],
                                           val_interval=rew_pred_training_params['val_interval'],
                                           ckpt_interval=rew_pred_training_params['ckpt_interval'])
        rpt_proc.join()
        ps_proc.terminate()
    elif general_params['mode'] == 'train_policy_with_original_rewards':
        env, a2c_proc = start_policy_training(cluster_dict=None,
                                              make_reward_predictor=None,
                                              gen_segments=False,
                                              start_policy_training_pipe=start_policy_training_flag,
                                              seg_pipe=seg_pipe,
                                              episode_vid_queue=episode_vid_queue,
                                              log_dir=general_params['log_dir'],
                                              a2c_params=a2c_params)
        start_policy_training_flag.put(True)
        a2c_proc.join()
        env.close()
    elif general_params['mode'] == 'train_policy_with_preferences':
        cluster_dict = create_cluster_dict(['ps', 'a2c', 'train'])
        ps_proc = start_parameter_server(cluster_dict, make_reward_predictor)
        env, a2c_proc = start_policy_training(cluster_dict=cluster_dict,
                                              make_reward_predictor=make_reward_predictor,
                                              gen_segments=True,
                                              start_policy_training_pipe=start_policy_training_flag,
                                              seg_pipe=seg_pipe,
                                              episode_vid_queue=episode_vid_queue,
                                              log_dir=general_params['log_dir'],
                                              a2c_params=a2c_params)
        pi_proc = start_pref_interface(seg_pipe=seg_pipe, pref_pipe=pref_pipe, log_dir=general_params['log_dir'],
                                       **pref_interface_params)
        rpt_proc = start_rew_pred_training(cluster_dict=cluster_dict,
                                           make_reward_predictor=make_reward_predictor,
                                           just_pretrain=False,
                                           pref_pipe=pref_pipe,
                                           start_policy_training_pipe=start_policy_training_flag,
                                           max_prefs=general_params['max_prefs'],
                                           prefs_dir=general_params['prefs_dir'],
                                           ckpt_path=None,
                                           n_initial_prefs=general_params['n_initial_prefs'],
                                           n_initial_epochs=rew_pred_training_params['n_initial_epochs'],
                                           val_interval=rew_pred_training_params['val_interval'],
                                           ckpt_interval=rew_pred_training_params['ckpt_interval'])
        a2c_proc.join()
        rpt_proc.terminate()
        pi_proc.terminate()
        ps_proc.terminate()
        env.close()
    else:
        raise Exception("Unknown mode: {}".format(general_params['mode']))

    if episode_renderer_proc:
        episode_renderer_proc.terminate()


def create_cluster_dict(jobs):
    ports = get_port_range(start_port=2200,
                           n_ports=len(jobs) + 1,
                           random_stagger=True)
    cluster_dict = {}
    for part, port in zip(jobs, ports):
        cluster_dict[part] = ['localhost:{}'.format(port)]
    return cluster_dict


def configure_a2c_logger(log_dir):
    a2c_dir = osp.join(log_dir, 'a2c')
    os.makedirs(a2c_dir)
    tb = logger.TensorBoardOutputFormat(a2c_dir)
    logger.Logger.CURRENT = logger.Logger(dir=a2c_dir, output_formats=[tb])


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


def get_initial_prefs(pref_pipe, n_initial_prefs, max_prefs):
    pref_db_val = PrefDB()
    pref_db_train = PrefDB()
    # Page 15: "We collect 500 comparisons from a randomly initialized policy
    # network at the beginning of training"
    while len(pref_db_train) < n_initial_prefs or len(pref_db_val) == 0:
        print("Waiting for preferences; %d so far" % len(pref_db_train))
        recv_prefs(pref_pipe, pref_db_train, pref_db_val, max_prefs)
        time.sleep(5.0)

    return pref_db_train, pref_db_val


def start_parameter_server(cluster_dict, make_reward_predictor):
    def f():
        rew_pred = make_reward_predictor('ps', cluster_dict)
        rew_pred.server.join()
    proc = Process(target=f, daemon=True)
    proc.start()
    return proc


def start_policy_training(cluster_dict,
                          make_reward_predictor,
                          gen_segments,
                          start_policy_training_pipe,
                          seg_pipe,
                          episode_vid_queue,
                          log_dir,
                          a2c_params):
    if a2c_params['env_id'] == 'MovingDotNoFrameskip-v0':
        policy_fn = MlpPolicy
    elif a2c_params['env_id'] == 'PongNoFrameskip-v4' or a2c_params['env_id'] == 'EnduroNoFrameskip-v4':
        policy_fn = CnnPolicy
    else:
        raise Exception("Unsure about policy network architecture for {}".format(a2c_params['env_id']))

    configure_a2c_logger(log_dir)

    misc_logs_dir = osp.join(log_dir, 'a2c_misc')
    os.makedirs(misc_logs_dir)

    # Done here because daemonic processes can't have children
    env = make_envs(a2c_params['env_id'], a2c_params['n_envs'], a2c_params['seed'])
    del a2c_params['env_id'], a2c_params['n_envs']

    def f():
        if make_reward_predictor:
            rew_pred = make_reward_predictor('a2c', cluster_dict)
        else:
            rew_pred = None
        easy_tf_log.set_dir(misc_logs_dir)
        learn(
            policy=policy_fn,
            env=env,
            seg_pipe=seg_pipe,
            start_policy_training_pipe=start_policy_training_pipe,
            episode_vid_queue=episode_vid_queue,
            reward_predictor=rew_pred,
            log_dir=log_dir,
            gen_segments=gen_segments,
            **a2c_params)
    proc = Process(target=f, daemon=True)
    proc.start()
    return env, proc


def start_pref_interface(seg_pipe, pref_pipe, max_segs, headless, log_dir):
    prefs_log_dir = osp.join(log_dir, 'pref_interface')
    os.makedirs(prefs_log_dir)
    def f():
        # The preference interface needs to get input from stdin.
        # stdin is automatically closed at the beginning of child processes in Python,
        # so this feels like a bit of a hack, but it seems to be fine.
        sys.stdin = os.fdopen(0)
        easy_tf_log.set_dir(prefs_log_dir)
        pi.run(seg_pipe=seg_pipe, pref_pipe=pref_pipe)
    # Needs to be done in the main process because does GUI work
    pi = PrefInterface(
        headless=headless, synthetic_prefs=headless, max_segs=max_segs)
    proc = Process(target=f, daemon=True)
    proc.start()
    return proc


def start_rew_pred_training(cluster_dict, make_reward_predictor, just_pretrain, pref_pipe,
                                    start_policy_training_pipe, max_prefs, n_initial_prefs, n_initial_epochs,
                                    prefs_dir, ckpt_path, val_interval, ckpt_interval):
    def f():
        rew_pred = make_reward_predictor('train', cluster_dict)
        rew_pred.init_network(ckpt_path)

        if prefs_dir is not None:
            pref_db_train, pref_db_val = load_pref_db(prefs_dir)
        else:
            pref_db_train, pref_db_val = get_initial_prefs(pref_pipe=pref_pipe,
                                                           n_initial_prefs=n_initial_prefs,
                                                           max_prefs=max_prefs)

        print("Pretraining reward predictor for {} epochs".format(n_initial_epochs))
        for i in range(n_initial_epochs):
            print("Epoch {}".format(i))
            rew_pred.train(pref_db_train, pref_db_val, val_interval)

            if i and i % ckpt_interval == 0:
                rew_pred.save()
        print("Reward predictor pretraining done")

        if just_pretrain:
            return

        start_policy_training_pipe.put(True)

        while True:
            rew_pred.train(pref_db_train, pref_db_val, val_interval)
            if i and i % ckpt_interval == 0:
                rew_pred.save()
            recv_prefs(pref_pipe, pref_db_train, pref_db_val, max_prefs)

    proc = Process(target=f, daemon=True)
    proc.start()
    return proc


def start_episode_renderer():
    def episode_vid_proc():
        vid_proc(episode_vid_queue,
                 playback_speed=2,
                 zoom_factor=2,
                 mode='play_through')
    episode_vid_queue = Queue()
    proc = Process(target=episode_vid_proc, args=(episode_vid_queue, ),
                   daemon=True).start()
    return episode_vid_queue, proc


if __name__ == '__main__':
    main()
