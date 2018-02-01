# Learning from Human Preferences

Side project, reproducing [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741).

Initially using custom A3C implementation (see [TensorFlow A3C](https://github.com/mrahtz/tensorflow-a3c)); now using OpenAI's A2C implementation.

In progress.

## Milestones

* 03/10/2017: Implemented reward predictor network
* 06/10/2017: Finished basic tests for reward predictor
* 17/10/2017: Implemented comparison interface
* 19/10/2017: Implemented simple 'moving dot' test environment
* 01/11/2017: Rewrote using multiple threads
* 11/11/2017: Rewrote reward predictor network to take larger batches
* 17/11/2017: Rewrote using processes instead of threads after discovering TensorFlow thread locking
* 23/11/2017: Implemented end-to-end tests for reward predictor
* ...a while spent fixing bugs...
* 01/02/2018: Partial success with training dot agent to move to the centre of a square (though he hasn't got the hang of moving horizontally yet?)

![](images/dot.gif)

Future targets:
* Implement remaining details (human error-adjusted softmax, label rate decay, etc.)
* Final goal: reproduce Enduro behaviour from <https://blog.openai.com/deep-reinforcement-learning-from-human-preferences/>

![](https://blog.openai.com/content/images/2017/06/enduro.gif)
