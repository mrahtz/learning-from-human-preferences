#!/usr/bin/env python3

import logging
import os
from os import path as osp
import queue
import sys
import time
from multiprocessing import Process, Queue

import cloudpickle
import easy_tf_log
import numpy as np

from a2c import logger
from a2c.a2c.a2c import learn
from a2c.a2c.policies import CnnPolicy, MlpPolicy
from a2c.common import set_global_seeds
from a2c.common.vec_env.subproc_vec_env import SubprocVecEnv
from params import parse_args
from pref_db import PrefDB
from pref_interface import PrefInterface
from reward_predictor import RewardPredictorEnsemble
from reward_predictor_core_network import net_cnn, net_moving_dot_features
from utils import VideoRenderer, get_port_range, profile_memory, make_env

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # filter out INFO messages


def main():
    general_params, a2c_params, \
        pref_interface_params, rew_pred_training_params = parse_args()

    if general_params['debug']:
        logging.getLogger().setLevel(logging.DEBUG)

    run(general_params,
        a2c_params,
        pref_interface_params,
        rew_pred_training_params)


def run(general_params,
        a2c_params,
        pref_interface_params,
        rew_pred_training_params):
    # See interprocess_communication_notes.txt
    seg_pipe = Queue(maxsize=200)
    pref_pipe = Queue(maxsize=300)
    start_policy_training_flag = Queue(maxsize=1)

    if general_params['render_episodes']:
        episode_vid_queue, episode_renderer = start_episode_renderer()
    else:
        episode_vid_queue = episode_renderer = None

    if a2c_params['env_id'] in ['MovingDot-v0', 'MovingDotNoFrameskip-v0']:
        reward_predictor_network = net_moving_dot_features
    elif a2c_params['env_id'] in ['PongNoFrameskip-v4', 'EnduroNoFrameskip-v4']:
        reward_predictor_network = net_cnn
    else:
        raise Exception("Unsure about reward predictor network for {}".format(
            a2c_params['env_id']))

    def make_reward_predictor(name, cluster_dict):
        return RewardPredictorEnsemble(
            cluster_job_name=name,
            cluster_dict=cluster_dict,
            log_dir=general_params['log_dir'],
            batchnorm=rew_pred_training_params['batchnorm'],
            dropout=rew_pred_training_params['dropout'],
            lr=rew_pred_training_params['lr'],
            core_network=reward_predictor_network)

    save_make_reward_predictor(general_params['log_dir'],
                               make_reward_predictor)

    if general_params['mode'] == 'gather_initial_prefs':
        env, a2c_proc = start_policy_training(
            cluster_dict=None,
            make_reward_predictor=None,
            gen_segments=True,
            start_policy_training_pipe=start_policy_training_flag,
            seg_pipe=seg_pipe,
            episode_vid_queue=episode_vid_queue,
            log_dir=general_params['log_dir'],
            a2c_params=a2c_params)
        pi, pi_proc = start_pref_interface(
            seg_pipe=seg_pipe,
            pref_pipe=pref_pipe,
            log_dir=general_params['log_dir'],
            **pref_interface_params)
        pref_db_train, pref_db_val = get_initial_prefs(
            pref_pipe=pref_pipe,
            n_initial_prefs=general_params['n_initial_prefs'],
            max_prefs=general_params['max_prefs'])
        train_path = osp.join(general_params['log_dir'], 'train_initial.pkl.gz')
        pref_db_train.save(train_path)
        print("Saved training preferences to '{}'".format(train_path))
        val_path = osp.join(general_params['log_dir'], 'val_initial.pkl.gz')
        pref_db_val.save(val_path)
        print("Saved validation preferences to '{}'".format(val_path))
        pi_proc.terminate()
        pi.stop_renderer()
        a2c_proc.terminate()
        env.close()
    elif general_params['mode'] == 'pretrain_reward_predictor':
        cluster_dict = create_cluster_dict(['ps', 'train'])
        ps_proc = start_parameter_server(cluster_dict, make_reward_predictor)
        rpt_proc = start_reward_predictor_training(
            cluster_dict=cluster_dict,
            make_reward_predictor=make_reward_predictor,
            just_pretrain=True,
            pref_pipe=pref_pipe,
            start_policy_training_pipe=start_policy_training_flag,
            max_prefs=general_params['max_prefs'],
            prefs_dir=general_params['prefs_dir'],
            load_ckpt_dir=None,
            n_initial_prefs=general_params['n_initial_prefs'],
            n_initial_epochs=rew_pred_training_params['n_initial_epochs'],
            val_interval=rew_pred_training_params['val_interval'],
            ckpt_interval=rew_pred_training_params['ckpt_interval'])
        rpt_proc.join()
        ps_proc.terminate()
    elif general_params['mode'] == 'train_policy_with_original_rewards':
        env, a2c_proc = start_policy_training(
            cluster_dict=None,
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
        env, a2c_proc = start_policy_training(
            cluster_dict=cluster_dict,
            make_reward_predictor=make_reward_predictor,
            gen_segments=True,
            start_policy_training_pipe=start_policy_training_flag,
            seg_pipe=seg_pipe,
            episode_vid_queue=episode_vid_queue,
            log_dir=general_params['log_dir'],
            a2c_params=a2c_params)
        m1 = profile_memory(general_params['log_dir'] + '/mem_a2c.log',
                            a2c_proc.pid)
        pi, pi_proc = start_pref_interface(
            seg_pipe=seg_pipe,
            pref_pipe=pref_pipe,
            log_dir=general_params['log_dir'],
            **pref_interface_params)
        m2 = profile_memory(general_params['log_dir'] + '/mem_pi.log',
                            pi_proc.pid)
        rpt_proc = start_reward_predictor_training(
            cluster_dict=cluster_dict,
            make_reward_predictor=make_reward_predictor,
            just_pretrain=False,
            pref_pipe=pref_pipe,
            start_policy_training_pipe=start_policy_training_flag,
            max_prefs=general_params['max_prefs'],
            prefs_dir=general_params['prefs_dir'],
            load_ckpt_dir=rew_pred_training_params['load_ckpt_dir'],
            n_initial_prefs=general_params['n_initial_prefs'],
            n_initial_epochs=rew_pred_training_params['n_initial_epochs'],
            val_interval=rew_pred_training_params['val_interval'],
            ckpt_interval=rew_pred_training_params['ckpt_interval'])
        m3 = profile_memory(general_params['log_dir'] + '/mem_rpt.log',
                            rpt_proc.pid)

        a2c_proc.join()
        m1.terminate()
        m2.terminate()
        m3.terminate()
        rpt_proc.terminate()
        pi_proc.terminate()
        pi.stop_renderer()
        ps_proc.terminate()
        env.close()
    else:
        raise Exception("Unknown mode: {}".format(general_params['mode']))

    if episode_renderer:
        episode_renderer.stop()


def save_make_reward_predictor(log_dir, make_reward_predictor):
    save_dir = osp.join(log_dir, 'reward_predictor_checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    with open(osp.join(save_dir, 'make_reward_predictor.pkl'), 'wb') as fh:
        fh.write(cloudpickle.dumps(make_reward_predictor))


def create_cluster_dict(jobs):
    n_ports = len(jobs) + 1
    ports = get_port_range(start_port=2200,
                           n_ports=n_ports,
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
    def wrap_make_env(env_id, rank):
        def _thunk():
            return make_env(env_id, seed + rank)
        return _thunk
    set_global_seeds(seed)
    env = SubprocVecEnv(env_id, [wrap_make_env(env_id, i)
                                 for i in range(n_envs)])
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
        make_reward_predictor('ps', cluster_dict)
        while True:
            time.sleep(1.0)

    proc = Process(target=f, daemon=True)
    proc.start()
    return proc


def start_policy_training(cluster_dict, make_reward_predictor, gen_segments,
                          start_policy_training_pipe, seg_pipe,
                          episode_vid_queue, log_dir, a2c_params):
    env_id = a2c_params['env_id']
    if env_id in ['MovingDotNoFrameskip-v0', 'MovingDot-v0']:
        policy_fn = MlpPolicy
    elif env_id in ['PongNoFrameskip-v4', 'EnduroNoFrameskip-v4']:
        policy_fn = CnnPolicy
    else:
        msg = "Unsure about policy network architecture for {}".format(
            a2c_params['env_id'])
        raise Exception(msg)

    configure_a2c_logger(log_dir)

    # Done here because daemonic processes can't have children
    env = make_envs(a2c_params['env_id'],
                    a2c_params['n_envs'],
                    a2c_params['seed'])
    del a2c_params['env_id'], a2c_params['n_envs']

    ckpt_dir = osp.join(log_dir, 'policy_checkpoints')
    os.makedirs(ckpt_dir)

    def f():
        if make_reward_predictor:
            reward_predictor = make_reward_predictor('a2c', cluster_dict)
        else:
            reward_predictor = None
        misc_logs_dir = osp.join(log_dir, 'a2c_misc')
        easy_tf_log.set_dir(misc_logs_dir)
        learn(
            policy=policy_fn,
            env=env,
            seg_pipe=seg_pipe,
            start_policy_training_pipe=start_policy_training_pipe,
            episode_vid_queue=episode_vid_queue,
            reward_predictor=reward_predictor,
            ckpt_save_dir=ckpt_dir,
            gen_segments=gen_segments,
            **a2c_params)

    proc = Process(target=f, daemon=True)
    proc.start()
    return env, proc


def start_pref_interface(seg_pipe, pref_pipe, max_segs, synthetic_prefs,
                         log_dir):
    def f():
        # The preference interface needs to get input from stdin. stdin is
        # automatically closed at the beginning of child processes in Python,
        # so this is a bit of a hack, but it seems to be fine.
        sys.stdin = os.fdopen(0)
        pi.run(seg_pipe=seg_pipe, pref_pipe=pref_pipe)

    # Needs to be done in the main process because does GUI setup work
    prefs_log_dir = osp.join(log_dir, 'pref_interface')
    pi = PrefInterface(synthetic_prefs=synthetic_prefs,
                       max_segs=max_segs,
                       log_dir=prefs_log_dir)
    proc = Process(target=f, daemon=True)
    proc.start()
    return pi, proc


def start_reward_predictor_training(cluster_dict,
                                    make_reward_predictor,
                                    just_pretrain,
                                    pref_pipe,
                                    start_policy_training_pipe,
                                    max_prefs,
                                    n_initial_prefs,
                                    n_initial_epochs,
                                    prefs_dir,
                                    load_ckpt_dir,
                                    val_interval,
                                    ckpt_interval):
    def f():
        rew_pred = make_reward_predictor('train', cluster_dict)
        rew_pred.init_network(load_ckpt_dir)

        if prefs_dir is not None:
            train_path = osp.join(prefs_dir, 'train_initial.pkl.gz')
            pref_db_train = PrefDB.load(train_path)
            print("Loaded training preferences from '{}'".format(train_path))
            n_prefs, n_segs = len(pref_db_train), len(pref_db_train.segments)
            print("({} preferences, {} segments)".format(n_prefs, n_segs))

            val_path = osp.join(prefs_dir, 'val_initial.pkl.gz')
            pref_db_val = PrefDB.load(val_path)
            print("Loaded validation preferences from '{}'".format(val_path))
            n_prefs, n_segs = len(pref_db_val), len(pref_db_val.segments)
            print("({} preferences, {} segments)".format(n_prefs, n_segs))
        else:
            pref_db_train, pref_db_val = get_initial_prefs(
                pref_pipe=pref_pipe,
                n_initial_prefs=n_initial_prefs,
                max_prefs=max_prefs)

        if not load_ckpt_dir:
            print("Pretraining reward predictor for {} epochs".format(
                n_initial_epochs))
            for i in range(n_initial_epochs):
                print("Reward predictor training epoch {}".format(i))
                rew_pred.train(pref_db_train, pref_db_val, val_interval)
                if i and i % ckpt_interval == 0:
                    rew_pred.save()
            print("Reward predictor pretraining done")
            rew_pred.save()

        if just_pretrain:
            return

        start_policy_training_pipe.put(True)
        
        i = 0
        while True:
            rew_pred.train(pref_db_train, pref_db_val, val_interval)
            if i and i % ckpt_interval == 0:
                rew_pred.save()
            recv_prefs(pref_pipe, pref_db_train, pref_db_val, max_prefs)

    proc = Process(target=f, daemon=True)
    proc.start()
    return proc


def start_episode_renderer():
    episode_vid_queue = Queue()
    renderer = VideoRenderer(
        episode_vid_queue,
        playback_speed=2,
        zoom=2,
        mode=VideoRenderer.play_through_mode)
    return episode_vid_queue, renderer


def recv_prefs(pref_pipe, pref_db_train, pref_db_val, max_prefs):
    """
    Get preferences from pref_pipe until there are none left to get.
    """
    val_fraction = 0.2
    n_recvd = 0
    # See interprocess_communication_notes.txt
    max_recv = 300
    while n_recvd < max_recv:
        try:
            s1, s2, mu = pref_pipe.get(block=True, timeout=1)
        except queue.Empty:
            break

        if np.random.rand() < val_fraction:
            pref_db_val.append(s1, s2, mu)
        else:
            pref_db_train.append(s1, s2, mu)

        if len(pref_db_val) > max_prefs * val_fraction:
            pref_db_val.del_first()
        assert len(pref_db_val) <= max_prefs * val_fraction

        if len(pref_db_train) > max_prefs * (1 - val_fraction):
            pref_db_train.del_first()
        assert len(pref_db_train) <= max_prefs * (1 - val_fraction)

        n_recvd += 1


if __name__ == '__main__':
    main()
