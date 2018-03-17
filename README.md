# Learning from Human Preferences

Reproduction of [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741).

Code is yet to be cleaned up, so expect mess.


## Results

* Successful training in a simple moving dot environment using synthetic preferences.

![](images/dot_success.gif)

* Successful training of Pong using synthetic preferences.

![](images/pong.gif)

* Reproduction of Enduro behaviour shown in [OpenAI's blog post](https://blog.openai.com/deep-reinforcement-learning-from-human-preferences/) using human preferences.

![](images/enduro.gif)


## Setup

To set up an isolated environment and install dependencies, install
[Pipenv](https://github.com/pypa/pipenv), then just run:

`pipenv install`

However, note that the correct version of TensorFlow must be installed
manually. Either:

`pipenv run pip install tensorflow==1.6.0`

or

`pipenv run pip install tensorflow-gpu==1.6.0`

depending on whether you have a GPU.

If you want to run tests, also run:

`pipenv install --dev`

Finally, before running any of the scripts, enter the environment with:

`pipenv shell`


## Code credits

A2C code in [`a2c`](a2c) is based on the implementation from [OpenAI's baselines](https://github.com/openai/baselines), commit [`f8663ea`](https://github.com/openai/baselines/commit/f8663ea).
