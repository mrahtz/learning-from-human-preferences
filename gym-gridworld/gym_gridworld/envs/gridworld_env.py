import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class ALE:
    def __init__(self):
        self.lives = lambda: 0


class GridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.size = [2, 2]
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(210, 160, 3))
        self.centre = np.array([80, 105])
        self.action_space = spaces.Discrete(5)
        self.viewer = None

        self.ale = ALE()
        seed = None
        self.np_random, seed1 = seeding.np_random(seed)
        self.random_start = True
        self.maxsteps = 1000

        self._reset()

    def get_action_meanings(self):
        return ['NOOP', 'DOWN', 'RIGHT', 'UP', 'LEFT']

    def _get_ob(self):
        ob = np.zeros((210, 160, 3), dtype=np.uint8)
        x = self.pos[0]
        y = self.pos[1]
        w = self.size[0]
        h = self.size[1]
        ob[y-h:y+h, x-w:x+w, :] = 255
        return ob

    def _step(self, action):
        assert action >= 0 and action <= 4

        self.prev_pos = self.pos[:]

        if action == 0:
            # NOOP
            pass
        elif action == 1:
            self.pos[1] += 1
        elif action == 2:
            self.pos[0] += 1
        elif action == 3:
            self.pos[1] -= 1
        elif action == 4:
            self.pos[0] -= 1
        self.pos[0] = np.clip(self.pos[0], self.size[0], 159 - self.size[0])
        self.pos[1] = np.clip(self.pos[1], self.size[1], 209 - self.size[1])

        ob = self._get_ob()

        self.steps += 1
        if self.steps < self.maxsteps:
            episode_over = False
        else:
            episode_over = True

        dist1 = np.linalg.norm(self.prev_pos - self.centre)
        dist2 = np.linalg.norm(self.pos - self.centre)
        if dist2 < dist1:
            reward = 1
        elif dist2 == dist1:
            reward = 0
        else:
            reward = -1

        return ob, reward, episode_over, {}

    def _reset(self):
        if self.random_start:
            x = np.random.randint(low=0, high=160)
            y = np.random.randint(low=0, high=210)
            self.pos = [x, y]
        else:
            self.pos = [0, 0]
        self.steps = 0
        ob = self._get_ob()
        return ob

    def _render(self, mode='human', close=False):
        # Ripped from gym's atari_env.py
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self._get_ob()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
