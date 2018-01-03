
import numpy as np

def predict_reward_frame(frame):
    middle = [84/2, 84/2]
    dot_coords = np.unravel_index(frame.argmin(), frame.shape)
    dot_coords = np.array(dot_coords)
    d = np.linalg.norm(dot_coords - middle)
    return -d


def predict_reward(segment):
    r_sum = 0
    for frame in segment:
        frame = frame[:, :, 0]
        r_sum += predict_reward_frame(frame)
    return r_sum


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
