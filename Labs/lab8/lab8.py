# MIT 6.034 Lab 8: Bayesian Inference
# Written by 6.034 staff

from nets import *

#### Part 1: Warm-up; Ancestors, Descendents, and Non-descendents ##############

def get_ancestors(net, var):
    "Return a set containing the ancestors of var"
    if not net.get_parents(var):
        return set()
    else:
        parents = set()
        for i in net.get_parents(var):
            parents.add(i)
            parents.update(get_ancestors(net,i))
        return parents

def get_descendants(net, var):
    "Returns a set containing the descendants of var"
    if not net.get_children(var):
        return set()
    else:
        children = set()
        for i in net.get_children(var):
            children.add(i)
            children.update(get_descendants(net,i))
        return children

def get_nondescendants(net, var):
    "Returns a set containing the non-descendants of var"
    non_descendants = set()
    descendants = get_descendants(net, var)
    for i in net.get_variables():
        if i not in descendants:
            non_descendants.add(i)
    non_descendants.remove(var)
    return non_descendants


#### Part 2: Computing Probability #############################################

def simplify_givens(net, var, givens):
    """
    If givens include every parent of var and no descendants, returns a
    simplified list of givens, keeping only parents.  Does not modify original
    givens.  Otherwise, if not all parents are given, or if a descendant is
    given, returns original givens.
    """
    if givens is None:
        return givens
    parents = net.get_parents(var)
    descendants = get_descendants(net,var)
    non_descendants = get_nondescendants(net,var)
    to_remove = non_descendants - parents
    givens_names = list(givens.keys())
    #print("given_names" + str(givens_names))
    #print("descendants" + str(descendants))
    flag = True
    for p in parents:
        if p not in givens_names:
            flag = False
    for d in descendants:
        if d in givens_names:
            flag = False
    #print("flag:" + str(flag))
    if flag:
        #print(givens.items())
        new_givens = {}
        for key,value in givens.items():
            if key not in to_remove:
                new_givens.update({key:value})
        return new_givens
    else:
        return givens

def probability_lookup(net, hypothesis, givens=None):
    "Looks up a probability in the Bayes net, or raises LookupError"

    #net.CPT_print(list(hypothesis.keys())[0])

    if givens is None:
        try:
            p1 = net.get_probability(hypothesis,givens)
        except ValueError:
            raise LookupError
        return p1
    else:
        try:
            givens = simplify_givens(net, list(hypothesis.keys())[0], givens)
            p2 = net.get_probability(hypothesis,givens)
        except ValueError:
            raise LookupError
        return p2


def probability_joint(net, hypothesis):
    "Uses the chain rule to compute a joint probability"

    ordered_vars = net.topological_sort()
    givens = {}
    p = 1 # initial probability
    for var in ordered_vars:
        if var in hypothesis:
            p = p * probability_lookup(net,{var: hypothesis[var]},givens)
            givens[var] = hypothesis[var]
    return p

def probability_marginal(net, hypothesis):
    "Computes a marginal probability as a sum of joint probabilities"

    jp = 0
    jp_table = net.combinations(net.get_variables(),hypothesis)
    for p in jp_table:
        jp = jp + probability_joint(net,p)
    return jp


def probability_conditional(net, hypothesis, givens=None):
    "Computes a conditional probability as a ratio of marginal probabilities"


    for v in hypothesis:
        if givens is not None:
            if v in givens:
                if hypothesis[v] is not givens[v]:
                    return 0
    if givens == hypothesis:
        return 1

    if givens is not None:
        d = dict(hypothesis, **givens)
        return probability_marginal(net, d) / probability_marginal(net, givens)
    else:
        return probability_marginal(net, hypothesis) / probability_marginal(net, givens)


def probability(net, hypothesis, givens=None):
    "Calls previous functions to compute any probability"
    return probability_conditional(net,hypothesis,givens)

#### Part 3: Counting Parameters ###############################################

def number_of_parameters(net):
    """
    Computes the minimum number of parameters required for the Bayes net.
    """
    vars = net.get_variables()

    table_domain = []
    for v in vars:
        v_domain = len(net.get_domain(v)) - 1
        sum_table = v_domain
        parents_domain = []
        for p in net.get_parents(v):
            parents_domain.append(len(net.get_domain(p)))
        sum_table = sum_table * product(parents_domain)
        table_domain.append(sum_table)

    return sum(table_domain)


#### Part 4: Independence ######################################################

def is_independent(net, var1, var2, givens=None):
    """
    Return True if var1, var2 are conditionally independent given givens,
    otherwise False. Uses numerical independence.
    """
    #p1 = probability(net,var1,givens)
    #p2 = probability(net,var2,givens)

    for comb1 in net.get_domain(var1):
        for comb2 in net.get_domain(var2):
            if givens is not None:
                d = dict({var2: comb2}, **givens)
                p1 = probability(net, {var1: comb1}, d)
            else:
                p1 = probability(net, {var1: comb1}, {var2: comb2})
            p2 = probability(net, {var1: comb1}, givens)
            if not approx_equal(p1, p2):
                return False
    return True

def is_structurally_independent(net, var1, var2, givens=None):
    """
    Return True if var1, var2 are conditionally independent given givens,
    based on the structure of the Bayes net, otherwise False.
    Uses structural independence only (not numerical independence).
    """
    # draw ancestial graph
    # print(givens)
    if givens is not None:
        subnet_variables = [var1, var2] + list(givens.keys())
        new_subvariables = subnet_variables.copy()
        for v in subnet_variables:
            for a in get_ancestors(net,v):
                if a not in subnet_variables:
                    new_subvariables.append(a)
    else:
        subnet_variables = [var1, var2]
        new_subvariables = subnet_variables.copy()
        for v in subnet_variables:
            for a in get_ancestors(net, v):
                if a not in subnet_variables:
                    new_subvariables.append(a)


    subnet = net.subnet(new_subvariables)
    # moralize
    new_subvariables = subnet.topological_sort()

    for v in new_subvariables:
        p = subnet.get_parents(v)
        for i in p:
            for j in p:
                subnet = subnet.link(i,j)

    # disorient
    subnet.make_bidirectional()
    # print(subnet)
    # delete the givens
    if givens is not None:
        for g in list(givens.keys()):
            #print(g)
            subnet = subnet.remove_variable(g)
    #print(subnet)
    # read answer
    if (not subnet.find_path(var1, var2)) or (var1 not in subnet.get_variables()) or (var2 not in subnet.get_variables()):
        return True
    else:
        return False


#### SURVEY ####################################################################

NAME = "XK"
COLLABORATORS = ""
HOW_MANY_HOURS_THIS_LAB_TOOK = 10
WHAT_I_FOUND_INTERESTING = ""
WHAT_I_FOUND_BORING = ""
SUGGESTIONS = ""
