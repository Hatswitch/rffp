import numpy as np
import sys
from readfeat import read_features
from pathlib import Path
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score

RF_TREES = 100
def cv_score(mean, std):
    return "{:.1f}% +/- {:.1f}%".format(mean*100, std*200)


def select_features(features, targets, goal=100, step=100):
    """
    :param features:
    :param targets:
    :return:
    Returns a transformed feature vector plus a list of features that were selected
    """

    print("Selecting best {} features".format(goal))
    rfe = RFE(RandomForestClassifier(RF_TREES), goal, step=step, verbose=True)

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

    clf = RandomForestClassifier(RF_TREES)

    top_features, mapping = select_features(features, targets)

    print("Selected features: {}".format(mapping))

    scores = np.array(list(kfold_scores(top_features, targets, clf)))
    print("Top feature Accuracy: {:.2f}% +/- {:.2f}%".format(scores.mean()*100, scores.std()*200))
    scores = np.array(list(kfold_scores(features, targets, clf)))
    print("All feature accuracy: {:.2f}% +/- {:.2f}%".format(scores.mean()*100, scores.std()*200))
