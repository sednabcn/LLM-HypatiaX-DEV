import numpy as np


def generate_data(func, var_ranges, n=100, extrapolation=False):
    """
    func: callable
    var_ranges: dict {var: (min, max)}
    extrapolation: if True → go outside training range
    """

    data = []

    for _ in range(n):
        sample = {}

        for var, (low, high) in var_ranges.items():
            if extrapolation:
                # go OUTSIDE training distribution
                width = high - low
                val = np.random.uniform(high + width, high + 2 * width)
            else:
                val = np.random.uniform(low, high)

            sample[var] = val

        try:
            y = func(**sample)
            sample["y"] = y
            data.append(sample)
        except Exception:
            continue

    return data
