import numpy as np


def feature_parse(x):
    if x == "'X'":
        return -1.
    else:
        return float(x)


def read_features(path):
    """
    Returns three matrices:
    :param path:
    :return:
        features (X): nsamples x nfeatures
        targets (y): nsamples x 1
        runs: nsamples x 1
    """
    features = []
    targets = []
    runs = []

    for fn in (path / "batch").glob("*f"):
        target, run = fn.name[:-1].split('-')
        with fn.open() as fp:
            raw_features = fp.readline().split()
        feature_row = [feature_parse(x) for x in raw_features]
        features.append(feature_row)
        targets.append(int(target))
        runs.append(int(run))

    return np.array(features), np.array(targets), np.array(runs)
