import queue
import random
import socket
import time
from multiprocessing import Process

import memory_profiler
import numpy as np
import pyglet
import tensorflow as tf

from scipy.ndimage import zoom


# https://github.com/joschu/modular_rl/blob/master/modular_rl/running_stat.py
# http://www.johndcook.com/blog/standard_deviation/
class RunningStat(object):
    def __init__(self, shape=()):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)

    def push(self, x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM)/self._n
            self._S[...] = self._S + (x - oldM)*(x - self._M)

    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        if self._n >= 2:
            return self._S/(self._n - 1)
        else:
            return np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape


class Im(object):
    def __init__(self, display=None):
        self.window = None
        self.isopen = False
        self.display = display

    def imshow(self, arr):
        if self.window is None:
            height, width = arr.shape
            self.window = pyglet.window.Window(
                width=width, height=height, display=self.display)
            self.width = width
            self.height = height
            self.isopen = True
        assert arr.shape == (
            self.height,
            self.width), "You passed in an image with the wrong number shape"
        image = pyglet.image.ImageData(
            self.width, self.height, 'L', arr.tobytes(), pitch=-self.width)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0, 0)
        self.window.flip()

    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False

    def __del__(self):
        self.close()


def get_most_recent_item(q):
    # Make sure we at least get something
    item = q.get(block=True)
    n_skipped = 0
    while True:
        try:
            item = q.get(block=True, timeout=0.1)
            n_skipped += 1
        except queue.Empty:
            break
    return item, n_skipped


class VideoRenderer:
    pause_cmd = "pause"

    def __init__(self, vid_queue, zoom_factor=1, playback_speed=1,
                 mode='restart_on_get'):
        assert mode == 'restart_on_get' or mode == 'play_through'
        self.mode = mode
        self.vid_queue = vid_queue
        self.zoom_factor = zoom_factor
        self.playback_speed = playback_speed
        self.proc = Process(target=self.render)
        self.proc.start()

    def stop(self):
        self.proc.terminate()

    def render(self):
        v = Im()
        frames = self.vid_queue.get(block=True)
        t = 0
        while True:
            # Add a dot showing progress
            width = frames[t].shape[1]
            fraction_played = t / len(frames)
            frames[t][-1][int(fraction_played * width)] = 255

            v.imshow(zoom(frames[t], self.zoom_factor))

            if self.mode == 'play_through':
                # Wait until having finished playing the current
                # set of frames. Then, stop, and get the most
                # recent set of frames.
                t += self.playback_speed
                if t >= len(frames):
                    frames, n_skipped = get_most_recent_item(self.vid_queue)
                    t = 0
                else:
                    time.sleep(1/60)
            elif self.mode == 'restart_on_get':
                # Always try and get a new set of frames to show.
                # If there is a new set of frames on the queue,
                # restart playback with those frames immediately.
                # Otherwise, just keep looping with the current frames.
                try:
                    item = self.vid_queue.get(block=False)
                    if item == VideoRenderer.pause_cmd:
                        frames = self.vid_queue.get(block=True)
                    else:
                        frames = item
                    t = 0
                except queue.Empty:
                    t = (t + self.playback_speed) % len(frames)
                    time.sleep(1/60)


def get_port_range(start_port, n_ports, random_stagger=False):
    # If multiple runs try and call this function at the same time,
    # the function could return the same port range.
    # To guard against this, automatically offset the port range.
    if random_stagger:
        start_port += random.randint(0, 20) * n_ports

    free_range_found = False
    while not free_range_found:
        ports = []
        for port_n in range(n_ports):
            port = start_port + port_n
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("127.0.0.1", port))
                ports.append(port)
            except socket.error as e:
                if e.errno == 98 or e.errno == 48:
                    print("Warning: port {} already in use".format(port))
                    break
                else:
                    raise e
            finally:
                s.close()
        if len(ports) < n_ports:
            # The last port we tried was in use
            # Try again, starting from the next port
            start_port = port + 1
        else:
            free_range_found = True

    return ports


def profile_memory(log_path, pid):
    def profile():
        with open(log_path, 'w') as f:
            # timeout=99999 is necesary because for external processes,
            # memory_usage otherwise defaults to only returning a single sample
            # Note that even with interval=1, because memory_profiler only
            # flushes every 50 lines, we still have to wait 50 seconds before
            # updates.
            memory_profiler.memory_usage(pid, stream=f,
                                         timeout=99999, interval=1)
    p = Process(target=profile, daemon=True)
    p.start()
    return p


def batch_iter(data, batch_size, shuffle=False):
    idxs = list(range(len(data)))
    if shuffle:
        np.random.shuffle(idxs)  # in-place

    start_idx = 0
    end_idx = 0
    while end_idx < len(data):
        end_idx = start_idx + batch_size
        if end_idx > len(data):
            end_idx = len(data)

        batch_idxs = idxs[start_idx:end_idx]
        batch = []
        for idx in batch_idxs:
            batch.append(data[idx])

        yield batch
        start_idx += batch_size


def get_dot_position(s):
    # s is (?, 84, 84, 4)
    s = s[..., -1]  # select last frame; now (?, 84, 84)

    x = tf.reduce_sum(s, axis=1)  # now (?, 84)
    x = tf.argmax(x, axis=1)

    y = tf.reduce_sum(s, axis=2)
    y = tf.argmax(y, axis=1)

    return x, y