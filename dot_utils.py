import numpy as np


def get_coords(frame):
    dot_coords = np.unravel_index(frame.argmin(), frame.shape)
    dot_coords = np.array(dot_coords)
    return dot_coords


def predict_reward_frame(frame):
    middle = [84/2, 84/2]
    dot_coords = get_coords(frame)
    d = np.linalg.norm(dot_coords - middle)
    return -d


def predict_reward(segment):
    r_sum = 0
    for frame in segment:
        frame = frame[:, :, 0]
        r_sum += predict_reward_frame(frame)
    return r_sum


def predict_action_rewards(segment):
    rewards = []
    for frame in segment:
        coords = get_coords(frame[:, :, -1])
        a = frame[0, 0, -1] - 100
        if a == 0:
            r = 0.0
        elif a == 1:
            r = np.sign(41.5 - coords[0])
        elif a == 2:
            r = np.sign(41.5 - coords[1])
        elif a == 3:
            r = np.sign(coords[0] - 41.5)
        elif a == 4:
            r = np.sign(coords[1] - 41.5)
        else:
            raise Exception("Unknown action {}".format(a))

        rewards.append(r)
    return rewards


def predict_preference(s1, s2):
    sums = []
    for s in [s1, s2]:
        sums.append(predict_reward(s))
    if sums[0] > sums[1]:
        predicted_mu = (1.0, 0.0)
    elif sums[0] == sums[1]:
        predicted_mu = (0.5, 0.5)
    else:
        predicted_mu = (0.0, 1.0)
    return predicted_mu


def predict_action_preference(s1, s2):
    sums = []
    for s in [s1, s2]:
        sums.append(sum(predict_action_rewards(s)))
    if sums[0] > sums[1]:
        predicted_mu = (1.0, 0.0)
    elif sums[0] == sums[1]:
        predicted_mu = (0.5, 0.5)
    else:
        predicted_mu = (0.0, 1.0)
    return predicted_mu
