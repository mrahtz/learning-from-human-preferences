from setuptools import setup

setup(name='gym_gridworld',
      version='1.0',
      packages=['gym_gridworld', 'gym_gridworld.envs'],
      install_requires=['gym', 'numpy']
)
