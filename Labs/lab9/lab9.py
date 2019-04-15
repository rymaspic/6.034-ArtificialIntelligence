# MIT 6.034 Lab 9: Boosting (Adaboost)
# Written by 6.034 staff

from math import log as ln
from utils import *


#### Part 1: Helper functions ##################################################

def initialize_weights(training_points):
    """Assigns every training point a weight equal to 1/N, where N is the number
    of training points.  Returns a dictionary mapping points to weights."""
    results = {}
    N = len(training_points)
    for p in training_points:
        results[p] = make_fraction(1/N)
    return results

def calculate_error_rates(point_to_weight, classifier_to_misclassified):
    """Given a dictionary mapping training points to their weights, and another
    dictionary mapping classifiers to the training points they misclassify,
    returns a dictionary mapping classifiers to their error rates."""
    result = {}
    #print(point_to_weight)
    #print(classifier_to_misclassified.values())
    for i in list(classifier_to_misclassified.keys()):
        sum = 0
        for j in classifier_to_misclassified[i]:
            sum = sum + point_to_weight[j]
        result[i] = make_fraction(sum)
    return result

def pick_best_classifier(classifier_to_error_rate, use_smallest_error=True):
    """Given a dictionary mapping classifiers to their error rates, returns the
    best* classifier, or raises NoGoodClassifiersError if best* classifier has
    error rate 1/2.  best* means 'smallest error rate' if use_smallest_error
    is True, otherwise 'error rate furthest from 1/2'."""
    #print(classifier_to_error_rate)
    if use_smallest_error:
        #print(list(classifier_to_error_rate.items()))
        best = min(sorted(list(classifier_to_error_rate.items())), key=lambda x: x[1])[0]
        #best = sorted(list(classifier_to_error_rate.items()), key=lambda x: x[1])[0]
        #print("best" + str(best))
        #print(calculate_error_rates[best])
        if make_fraction(classifier_to_error_rate[best]) == make_fraction(1/2):
            raise NoGoodClassifiersError
        else:
            #print(best)
            return best
   # elif not use_smallest_error:
    else:
        best = max(sorted(list(classifier_to_error_rate.items())), key=lambda x:
        abs(make_fraction(1/2) - make_fraction(x[1])))[0]
        if make_fraction(classifier_to_error_rate[best]) == make_fraction(1/2):
            raise NoGoodClassifiersError
        else:
            #print(best)
            return best

def calculate_voting_power(error_rate):
    """Given a classifier's error rate (a number), returns the voting power
    (aka alpha, or coefficient) for that classifier."""
    if error_rate is 0:
        return INF
    elif error_rate is 1:
        return -INF
    else:
        e = error_rate
        return ((1 / 2) * ln((1 - e) / e))

def get_overall_misclassifications(H, training_points, classifier_to_misclassified):
    """Given an overall classifier H, a list of all training points, and a
    dictionary mapping classifiers to the training points they misclassify,
    returns a set containing the training points that H misclassifies.
    H is represented as a list of (classifier, voting_power) tuples."""
    misclassified_points = []
    for p in training_points:
        sum = 0
        for classifier, vp in H:
            if p in classifier_to_misclassified[classifier]:
                sum = sum - vp
            else:
                sum = sum + vp
        if sum <= 0:
            misclassified_points.append(p)

    return set(misclassified_points)

def is_good_enough(H, training_points, classifier_to_misclassified, mistake_tolerance=0):
    """Given an overall classifier H, a list of all training points, a
    dictionary mapping classifiers to the training points they misclassify, and
    a mistake tolerance (the maximum number of allowed misclassifications),
    returns False if H misclassifies more points than the tolerance allows,
    otherwise True.  H is represented as a list of (classifier, voting_power)
    tuples."""
    misclassified_points = get_overall_misclassifications(H,training_points,classifier_to_misclassified)
    if len(misclassified_points)>mistake_tolerance:
        return False
    else:
        return True

def update_weights(point_to_weight, misclassified_points, error_rate):
    """Given a dictionary mapping training points to their old weights, a list
    of training points misclassified by the current weak classifier, and the
    error rate of the current weak classifier, returns a dictionary mapping
    training points to their new weights.  This function is allowed (but not
    required) to modify the input dictionary point_to_weight."""

    results = {}
    #print(error_rate)
    for p in point_to_weight.items():
        if p[0] in misclassified_points:
            results[p[0]] = make_fraction(1 / 2) * make_fraction(1, error_rate) * p[1]
        else:
            results[p[0]] = make_fraction(1 / 2) * make_fraction(1, 1 - error_rate) * p[1]
    return results


#### Part 2: Adaboost ##########################################################

def adaboost(training_points, classifier_to_misclassified,
             use_smallest_error=True, mistake_tolerance=0, max_rounds=INF):
    """Performs the Adaboost algorithm for up to max_rounds rounds.
    Returns the resulting overall classifier H, represented as a list of
    (classifier, voting_power) tuples."""

    H = []
    round = 0
    # 1.use the uniform weights to start
    points_to_weights = initialize_weights(training_points)
    while round < max_rounds:
        # 2.pick the classifier that has the lowest error rate
        error_rates = calculate_error_rates(points_to_weights,classifier_to_misclassified)
        try:
            best_classifier = pick_best_classifier(error_rates,use_smallest_error)
        except:
            return H
        # 3.use the best c to calculate the error rate associated with the step
        error_rate = make_fraction(error_rates[best_classifier])
        # 4.determine the alpha
        voting_power= calculate_voting_power(error_rate)
        H.append((best_classifier,voting_power))
        misclassified_points = classifier_to_misclassified[best_classifier]
        if is_good_enough(H,training_points,classifier_to_misclassified,mistake_tolerance):
            return H
        points_to_weights = update_weights(points_to_weights,misclassified_points,error_rate)
        round = round + 1

    return H


#### SURVEY ####################################################################

NAME = "XK"
COLLABORATORS = ""
HOW_MANY_HOURS_THIS_LAB_TOOK = 6
WHAT_I_FOUND_INTERESTING = ""
WHAT_I_FOUND_BORING = ""
SUGGESTIONS = ""
