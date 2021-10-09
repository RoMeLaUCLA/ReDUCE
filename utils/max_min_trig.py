import numpy as np

def find_min_max_cos(t1, t2):
    # Assume the range of t is -pi ~ pi
    if (t1 < 0 and t2 > 0) or (t1 > 0 and t2 < 0):
        max_cos = 1
    else:
        max_cos = max(np.cos(t1), np.cos(t2))

    min_cos = min(np.cos(t1), np.cos(t2))

    return min_cos, max_cos


def find_min_max_sin(t1, t2):
    # Assume the range of t is -pi ~ pi
    if (t1 < np.pi / 2 and t2 > np.pi / 2) or (t1 > np.pi / 2 and t2 < np.pi / 2):
        max_sin = 1
    else:
        max_sin = max(np.sin(t1), np.sin(t2))

    if (t1 < -np.pi / 2 and t2 > -np.pi / 2) or (t1 > -np.pi / 2 and t2 < -np.pi / 2):
        min_sin = -1
    else:
        min_sin = min(np.sin(t1), np.sin(t2))

    return min_sin, max_sin
