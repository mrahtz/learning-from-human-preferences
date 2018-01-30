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
    middle = [84/2, 84/2]
    old_coords = get_coords(segment[0][:, :, 0])
    old_d = np.linalg.norm(old_coords - middle)
    rewards = []
    for frame in segment[1:]:
        new_coords = get_coords(frame[:, :, 0])
        new_d = np.linalg.norm(new_coords - middle)
        if new_d < old_d:
            rewards.append(1.)
        elif new_d > old_d:
            rewards.append(-1.)
        else:
            rewards.append(0.)
        old_d = new_d
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
