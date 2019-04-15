# MIT 6.034 Lab 7: Support Vector Machines
# Written by 6.034 staff

from svm_data import *
from functools import reduce


#### Part 1: Vector Math #######################################################

def dot_product(u, v):
    """Computes the dot product of two vectors u and v, each represented 
    as a tuple or list of coordinates. Assume the two vectors are the
    same length."""
    dot = lambda X, Y: sum(map(lambda x, y: x * y, X, Y))
    return dot(u, v)

import math

def norm(v):
    """Computes the norm (length) of a vector v, represented 
    as a tuple or list of coords."""

    return math.sqrt(dot_product(v, v))

#### Part 2: Using the SVM Boundary Equations ##################################

def positiveness(svm, point):
    """Computes the expression (w dot x + b) for the given Point x."""
    pos = dot_product(svm.w, point) + svm.b

    return pos

def classify(svm, point):
    """Uses the given SVM to classify a Point. Assume that the point's true
    classification is unknown.
    Returns +1 or -1, or 0 if point is on boundary."""
    pos = positiveness(svm,point)
    if pos > 0:
        return 1
    elif pos < 0:
        return -1
    else:
        return 0

def margin_width(svm):
    """Calculate margin width based on the current boundary."""

    return 2/norm(svm.w)

def check_gutter_constraint(svm):
    """Returns the set of training points that violate one or both conditions:
        * gutter constraint (positiveness == classification, for support vectors)
        * training points must not be between the gutters
    Assumes that the SVM has support vectors assigned."""

    result = []
    for point in svm.training_points:
        if (-1 < positiveness(svm,point) < 1):
            result.append(point)
    for sv in svm.support_vectors:
        if not (sv.classification == positiveness(svm,sv)):
            result.append(sv)
    return set(result)

#### Part 3: Supportiveness ####################################################

def check_alpha_signs(svm):
    """Returns the set of training points that violate either condition:
        * all non-support-vector training points have alpha = 0
        * all support vectors have alpha > 0
    Assumes that the SVM has support vectors assigned, and that all training
    points have alpha values assigned."""

    #violate
    result = []

    for point in svm.training_points:
        if point in svm.support_vectors:
            if point.alpha <= 0:
                result.append(point)
        else:
            if not (point.alpha == 0):
                result.append(point)

    return set(result)

def check_alpha_equations(svm):
    """Returns True if both Lagrange-multiplier equations are satisfied,
    otherwise False. Assumes that the SVM has support vectors assigned, and
    that all training points have alpha values assigned."""

# vector_add(vec1, vec2)
# Given two vectors represented as iterable vectors (lists or tuples of coordinates) or Points,
# returns their vector sum as a list of coordinates.
# scalar_mult(c, vec)
# Given a constant scalar and an iterable vector (as a tuple or list of coordinates)
# or a Point, returns a scaled list of coordinates.
    sum = 0
    for point in svm.training_points:
        sum += point.classification * point.alpha
    if sum:
        return False
    sum = 0
    for point in svm.training_points:
        c = point.classification * point.alpha
        if sum == 0:
            sum = scalar_mult(c, point.coords)
        else:
            sum = vector_add(sum, scalar_mult(c, point.coords))
    if not (sum == svm.w):
        return False

    return True

#### Part 4: Evaluating Accuracy ###############################################

def misclassified_training_points(svm):
    """Returns the set of training points that are classified incorrectly
    using the current decision boundary."""
    result = []
    for point in svm.training_points:
        if not (classify(svm,point) == point.classification):
            result.append(point)


    return set(result)

#### Part 5: Training an SVM ###################################################

def update_svm_from_alphas(svm):
    """Given an SVM with training data and alpha values, use alpha values to
    update the SVM's support vectors, w, and b. Return the updated SVM."""

    for s in svm.support_vectors:
        if not s.alpha >0:
            svm.support_vectors.remove(s)

    #print(svm.support_vectors)
    for point in svm.training_points:
        if point.alpha > 0 and (point not in svm.support_vectors):
            svm.support_vectors.append(point)
    sum = 0
    for point in svm.training_points:
        c = point.classification * point.alpha
        if sum == 0:
            sum = scalar_mult(c, point.coords)
        else:
            sum = vector_add(sum, scalar_mult(c, point.coords))

    new_w =sum
    #print(sum)

    sv_pos = []
    sv_neg = []
    #print(svm.support_vectors)

    for sv in svm.support_vectors:
        if sv.classification == 1:
            sv_pos.append(sv)
        elif sv.classification == -1:
            sv_neg.append(sv)
    #print("tttttttttttttttt")
    #print(sv_pos)

    b_max = sv_pos[0].classification - dot_product(new_w,sv_pos[0].coords)

    for sv in sv_pos:
        b = sv.classification - dot_product(new_w,sv.coords)
        if b > b_max:
            b_max = b

    b_min = sv_neg[0].classification - dot_product(new_w,sv_neg[0].coords)

    for sv in sv_neg:
        b = sv.classification - dot_product(new_w, sv.coords)
        if b < b_min:
            b_min = b

    new_b = (b_max + b_min)/2

    svm.set_boundary(new_w, new_b)

    return svm

#### Part 6: Multiple Choice ###################################################

ANSWER_1 = 11
ANSWER_2 = 6
ANSWER_3 = 3
ANSWER_4 = 2

ANSWER_5 = ['A','D']
ANSWER_6 = ['A','B','D']
ANSWER_7 = ['A','B','D']
ANSWER_8 = []
ANSWER_9 = ['A','B','D']
ANSWER_10 = ['A','B','D']

ANSWER_11 = False
ANSWER_12 = True
ANSWER_13 = False
ANSWER_14 = False
ANSWER_15 = False
ANSWER_16 = True

ANSWER_17 = [1,3,6,8] #Q
ANSWER_18 = [1,2,4,5,6,7,8]
ANSWER_19 = [1,2,4,5,6,7,8]

ANSWER_20 = 6


#### SURVEY ####################################################################

NAME = "XK"
COLLABORATORS = ""
HOW_MANY_HOURS_THIS_LAB_TOOK = 9
WHAT_I_FOUND_INTERESTING = ""
WHAT_I_FOUND_BORING = ""
SUGGESTIONS = ""
