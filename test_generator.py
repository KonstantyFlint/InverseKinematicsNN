import random
from math import sin, cos, pi

import numpy as np


def get_coords(alpha: float, beta: float):
    return sin(alpha) - sin(alpha + beta), cos(alpha) - cos(alpha + beta)


def make_normalizer(from_min, from_max, to_min, to_max):
    from_range = from_max - from_min
    to_range = to_max - to_min

    def normalize(x):
        return (x - from_min) * to_range / from_range + to_min

    return normalize


normalize_coord = make_normalizer(-2, 2, -1, 1)
normalize_angle = make_normalizer(0, pi, 0.1, 0.9)


def generate_test_case():
    alpha = random.uniform(0, pi)
    beta = random.uniform(0, pi)
    x, y = get_coords(alpha, beta)
    return (
        np.array([
            normalize_coord(x),
            normalize_coord(y),
        ]),
        np.array([
            normalize_angle(alpha),
            normalize_angle(beta),
        ])
    )


def generate_test_cases(num_cases):
    X = []
    y = []
    for _ in range(num_cases):
        coords, angles = generate_test_case()
        X.append(coords)
        y.append(angles)
    return np.array(X), np.array(y)
