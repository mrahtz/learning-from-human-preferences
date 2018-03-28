"""
An environment wrapper for Enduro which blanks out the speedometer (so that the
agent doesn't inadvertently learn reward-related information from it) and
signals 'done' once weather begins to change (so that the observations don't
change so much and therefore the reward predictor can learn more easily).
"""

from gym import Wrapper


class EnduroWrapper(Wrapper):
    def __init__(self, env):
        super(EnduroWrapper, self).__init__(env)
        assert str(env) == '<TimeLimit<AtariEnv<EnduroNoFrameskip-v4>>>'
        self._steps = None

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        # Blank out all the speedometer stuff
        observation[160:] = 0
        self._steps += 1
        # Done once the weather starts to change
        if self._steps == 3000:
            done = True
        return observation, reward, done, info

    def reset(self):
        self._steps = 0
        return self.env.reset()
