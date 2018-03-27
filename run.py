#!/usr/bin/env python3

import logging
import os
from os import path as osp
import sys
import time
from multiprocessing import Process, Queue

import cloudpickle
import easy_tf_log
from a2c import logger
from a2c.a2c.a2c import learn
from a2c.a2c.policies import CnnPolicy, MlpPolicy
from a2c.common import set_global_seeds
from a2c.common.vec_env.subproc_vec_env import SubprocVecEnv
from params import parse_args, PREFS_VAL_FRACTION
from pref_db import PrefDB, PrefBuffer
from pref_interface import PrefInterface
from reward_predictor import RewardPredictorEnsemble
from reward_predictor_core_network import net_cnn, net_moving_dot_features
from utils import VideoRenderer, get_port_range, make_env

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
    seg_pipe = Queue(maxsize=1)
    pref_pipe = Queue(maxsize=1)
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

        n_train = general_params['max_prefs'] * (1 - PREFS_VAL_FRACTION)
        n_val = general_params['max_prefs'] * PREFS_VAL_FRACTION
        pref_db_train = PrefDB(maxlen=n_train)
        pref_db_val = PrefDB(maxlen=n_val)
        pref_buffer = PrefBuffer(db_train=pref_db_train, db_val=pref_db_val)
        pref_buffer.start_recv_thread(pref_pipe)
        pref_buffer.wait_until_len(general_params['n_initial_prefs'])
        pref_db_train, pref_db_val = pref_buffer.get_dbs()

        train_path = osp.join(general_params['log_dir'], 'train_initial.pkl.gz')
        pref_db_train.save(train_path)
        print("Saved training preferences to '{}'".format(train_path))
        val_path = osp.join(general_params['log_dir'], 'val_initial.pkl.gz')
        pref_db_val.save(val_path)
        print("Saved validation preferences to '{}'".format(val_path))

        pi_proc.terminate()
        pi.stop_renderer()
        a2c_proc.terminate()
        pref_buffer.stop_recv_thread()

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
        pi, pi_proc = start_pref_interface(
            seg_pipe=seg_pipe,
            pref_pipe=pref_pipe,
            log_dir=general_params['log_dir'],
            **pref_interface_params)
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
        # We wait for A2C to complete the specified number of policy training
        # steps
        a2c_proc.join()
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
        msg = "Unsure about policy network for {}".format(a2c_params['env_id'])
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
            n_train = max_prefs * (1 - PREFS_VAL_FRACTION)
            n_val = max_prefs * PREFS_VAL_FRACTION
            pref_db_train = PrefDB(maxlen=n_train)
            pref_db_val = PrefDB(maxlen=n_val)

        pref_buffer = PrefBuffer(db_train=pref_db_train,
                                 db_val=pref_db_val)
        pref_buffer.start_recv_thread(pref_pipe)
        if prefs_dir is None:
            pref_buffer.wait_until_len(n_initial_prefs)

        if not load_ckpt_dir:
            print("Pretraining reward predictor for {} epochs".format(
                n_initial_epochs))
            pref_db_train, pref_db_val = pref_buffer.get_dbs()
            for i in range(n_initial_epochs):
                # Note that we deliberately don't update the preferences
                # databases during pretraining to keep the number of
                # fairly preferences small so that pretraining doesn't take too
                # long.
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
            pref_db_train, pref_db_val = pref_buffer.get_dbs()
            rew_pred.train(pref_db_train, pref_db_val, val_interval)
            if i and i % ckpt_interval == 0:
                rew_pred.save()

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


if __name__ == '__main__':
    main()
