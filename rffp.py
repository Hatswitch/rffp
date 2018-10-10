import numpy as np
import sys
from pathlib import Path
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

RF_TREES = 100

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

def select_features(features, targets, goal=100):
    """
    :param features:
    :param targets:
    :return:
    Returns a transformed feature vector plus a list of features that were selected
    """

    print("Selecting best {} features".format(goal))
    rfe = RFE(RandomForestClassifier(RF_TREES), goal, step=100, verbose=True)

    rfe.fit(features, targets)

    mapping = [i for i, x in enumerate(rfe.get_support()) if x]

    return rfe.transform(features), mapping

def kfold_scores(features, targets, clf, folds=10, shuffle=True):
    kf = KFold(n_splits=folds, shuffle=shuffle)

    for train, test in kf.split(features):
        clf.fit(features[train,:], targets[train])
        yield clf.score(features[test,:], targets[test])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = Path(".")

    features, targets, runs = read_features(path)

    print("Loaded {0} samples with {1} features".format(*features.shape))

    top_features, mapping = select_features(features, targets)

    print("Selected features: {}".format(mapping))

    clf = RandomForestClassifier(RF_TREES)

    scores = np.array(list(kfold_scores(top_features, targets, clf)))

    print("Accuracy: {:.2f}% +/- {:.2f}%".format(scores.mean()*100, scores.std()*200))