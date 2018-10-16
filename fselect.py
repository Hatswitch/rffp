import sys
from pathlib import Path
from readfeat import read_features
from rffp import select_features, RF_TREES
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from math import inf

def cv_score(mean, std):
    return "{:.1f}% +/- {:.1f}%".format(mean*100, std*200)


def score_features(feat_options, data, target, clf, num_cv=10):
    for newf in range(data.shape[1]):
        for oldf in feat_options:
            if newf in oldf:
                continue
            scores = cross_val_score(clf, data[:,oldf + [newf]], targets, cv=num_cv, n_jobs=-1)
            print("{} + {}: {}".format(oldf, newf, cv_score(scores.mean(), scores.std())))
            yield scores.mean(), scores.std(), oldf + [newf]

def expand_features(feat_options, data, target):
    scores = list(score_features(feat_options, data, target, RandomForestClassifier(RF_TREES)))
    best_mean, best_std, _= max(scores)
    for mean, std, fo in scores:
        if mean + std >= best_mean - best_std:
            yield mean, std, fo

def find_best_features(data, target, max_features=inf):
    cur_list = [(0,0,[])]
    prev_best = None

    while len(cur_list[0][2]) < max_features:
        new_list = list(expand_features([x[2] for x in cur_list], data, target))
        print("New list: {}, best: {}".format([x[2] for x in new_list], max(new_list)))
        best_mean, best_std, _ = max(new_list)
        if prev_best is not None and prev_best >= best_mean - best_std:
            # if new mean - new std < old_mean, we are done
            print("Stopping: {} {} {}".format(prev_best, best_mean, best_std))
            return cur_list
        cur_list = new_list

    return cur_list

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = Path(".")

    features, targets, runs = read_features(path)

    print("Loaded {0} samples with {1} features".format(*features.shape))

    if len(sys.argv) > 2:
        top20_mapping = [int(arg.strip(',')) for arg in sys.argv[2:]]
        top20_features = features[:,top20_mapping]
        print("Using features: {}".format(top20_mapping))
    else:
        print("Performing feature elimination")

        top100_features, top100_mapping = select_features(features, targets)

        print("Selected top 100 features: {}".format(top100_mapping))

        top20_features, top20_mapping = select_features(top100_features, targets, goal=20, step=1)

        top20_mapping = [top100_mapping[i] for i in top20_mapping]

        print("Selected top 20 features: {}".format(top20_mapping))

    best = find_best_features(top20_features, targets)

    print("Final results")
    for mean, std, feats in best:
        print("{}: {}".format([top20_mapping[i] for i in feats], cv_score(mean, std)))