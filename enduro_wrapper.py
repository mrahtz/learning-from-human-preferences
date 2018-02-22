from gym import Wrapper


class EnduroWrapper(Wrapper):
    def __init__(self, env):
        super(EnduroWrapper, self).__init__(env)
        assert str(env) == '<TimeLimit<AtariEnv<EnduroNoFrameskip-v4>>>'
        self._steps = None

    def step(self, action):
        assert self._steps is not None, \
            "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        # Blank out speedometer etc.
        observation[160:] = 0
        self._steps += 1
        if self._steps == 3000:
            done = True
        return observation, reward, done, info

    def reset(self):
        self._steps = 0
        return self.env.reset()
