# MIT 6.034 Lab 4: Constraint Satisfaction Problems
# Written by 6.034 staff

from constraint_api import *
from test_problems import get_pokemon_problem


#### Part 1: Warmup ############################################################

def has_empty_domains(csp) :
    """Returns True if the problem has one or more empty domains, otherwise False"""
    for var in csp.get_all_variables():
        if not csp.get_domain(var):
            return True
    else:
        return False


def check_all_constraints(csp) :
    """Return False if the problem's assigned values violate some constraint,
    otherwise True"""

    values = csp.assignments
    # Check every variable against every other variable
    for var1 in values:
        for var2 in values:
            # Through every constraint that applies to the two variables
            for constraint in csp.constraints_between(var1, var2):
                # If the constraint fails
                if not constraint.check(values[var1], values[var2]):
                    # Return False
                    return False
    # Otherwise return True, everything passed
    return True


#### Part 2: Depth-First Constraint Solver #####################################

def solve_constraint_dfs(problem):
    """
    Solves the problem using depth-first search.  Returns a tuple containing:
    1. the solution (a dictionary mapping variables to assigned values)
    2. the number of extensions made (the number of problems popped off the agenda).
    If no solution was found, return None as the first element of the tuple.
    """
    extensions = 0
    agenda = [problem]
    while agenda:
        first_problem = agenda.pop()
        #print(first_problem)
        extensions = extensions + 1
        if has_empty_domains(first_problem):
            continue
        else:
            if not check_all_constraints(first_problem):
                #print("hhhhhhhhhhhhhh")
                continue
        #print(unassigned_variables)
            else:
                unassigned_variables = first_problem.unassigned_vars
                #print(unassigned_variables)
                if unassigned_variables == []:
                    solution = first_problem.assignments
                    return (solution, extensions)
                     #break #solution found
                else:
                    next_unassigned_variable = first_problem.pop_next_unassigned_var()
                    #print(next_unassigned_variable)
                    values_in_domain = first_problem.get_domain(next_unassigned_variable)
                    new_problem_lists = []
                    for val in values_in_domain:
                        new_problem = first_problem.copy().set_assignment(next_unassigned_variable, val)

                        new_problem_lists.append(new_problem)
                    #print(agenda)
                    new_problem_lists.reverse()
                    agenda = agenda + new_problem_lists
                    # print("HHHHHHHHH")
                    # print(agenda)
    return (None,extensions)

# QUESTION 1: How many extensions does it take to solve the Pokemon problem
#    with DFS?

#print(solve_constraint_dfs(get_pokemon_problem()))

# Hint: Use get_pokemon_problem() to get a new copy of the Pokemon problem
#    each time you want to solve it with a different search method.

ANSWER_1 = solve_constraint_dfs(get_pokemon_problem())[1]


#### Part 3: Forward Checking ##################################################


def check_violation(csp,var1,var2,val1,val2):
    for cons in csp.constraints_between(var1,var2):
        if not cons.check(val1,val2):
            return True # True: there exists a violation between val1 and val2
    return False # False: no vilation


def eliminate_from_neighbors(csp, var) :
    """
    Eliminates incompatible values from var's neighbors' domains, modifying
    the original csp.  Returns an alphabetically sorted list of the neighboring
    variables whose domains were reduced, with each variable appearing at most
    once.  If no domains were reduced, returns empty list.
    If a domain is reduced to size 0, quits immediately and returns None.
    """

    result = []
    var_domain = csp.get_domain(var)
    neibor_vars = csp.get_neighbors(var)
    for W in neibor_vars: # W is the neighboring variables
        flag_reduced = False #if true then the neibor_var is reduced
        values_to_reduce = []
        for w in csp.get_domain(W): # w is the values of the W
            flag_violation=[]# store every result of the constriant test of w and v
            for v in var_domain:
                if check_violation(csp,var,W,v,w):
                    flag_violation.append(True)
                else:
                    flag_violation.append(False)
            #print(flag_violation)
            flag_to_reduce = True #if True then the w should be removed from the W
            for flag in flag_violation:
                if not flag:
                    flag_to_reduce = False
                    break
            if flag_to_reduce:
                #print(flag_volation)
                values_to_reduce.append(w)
                #print(csp.get_domain(W))
                flag_reduced = True
        for val in values_to_reduce:
            csp.eliminate(W,val)
        if not csp.get_domain(W):
            return None
        if flag_reduced:
            #print(W)
            if W not in result:
                result.append(W)

   # result = list(set(result))
    return result

# Because names give us power over things (you're free to use this alias)
forward_check = eliminate_from_neighbors

def solve_constraint_forward_checking(problem):
    """
    Solves the problem using depth-first search with forward checking.
    Same return type as solve_constraint_dfs.
    """
    extensions = 0
    agenda = [problem]
    while agenda:

        first_problem = agenda.pop()
        # print(first_problem)
        extensions = extensions + 1
        if has_empty_domains(first_problem):
            continue
        else:
            if not check_all_constraints(first_problem):
                continue
                # print(unassigned_variables)
            else:
                #forward_check(first_problem, next_unassigned_variable)

                unassigned_variables = first_problem.unassigned_vars
                # print(unassigned_variables)
                if unassigned_variables == []:
                    solution = first_problem.assignments
                    return (solution, extensions)
                    # break #solution found
                else:
                    next_unassigned_variable = first_problem.pop_next_unassigned_var()
                    # print(next_unassigned_variable)
                    values_in_domain = first_problem.get_domain(next_unassigned_variable)
                    new_problem_lists = []
                    for val in values_in_domain:
                        new_problem = first_problem.copy().set_assignment(next_unassigned_variable, val)
                        forward_check(new_problem,next_unassigned_variable)
                        new_problem_lists.append(new_problem)
                    # print(agenda)
                    new_problem_lists.reverse()
                    agenda = agenda + new_problem_lists
                    # print("HHHHHHHHH")
                    # print(agenda)

    return (None, extensions)

# QUESTION 2: How many extensions does it take to solve the Pokemon problem
#    with DFS and forward checking?

ANSWER_2 = solve_constraint_forward_checking(get_pokemon_problem())[1]


#### Part 4: Domain Reduction ##################################################

def domain_reduction(csp, queue=None) :
    """
    Uses constraints to reduce domains, propagating the domain reduction
    to all neighbors whose domains are reduced during the process.
    If queue is None, initializes propagation queue by adding all variables in
    their default order. 
    Returns a list of all variables that were dequeued, in the order they
    were removed from the queue.  Variables may appear in the list multiple times.
    If a domain is reduced to size 0, quits immediately and returns None.
    This function modifies the original csp.
    """
    dequeue = []
    if queue is None:
        queue = csp.get_all_variables()
    while queue:
        var = queue.pop(0)
        dequeue.append(var)
        reduced_neighbors = eliminate_from_neighbors(csp,var)
        #print(reduced_neighbors)
        if reduced_neighbors is None:
            return None
        for neighbor in reduced_neighbors:
            if neighbor not in queue:
                queue.append(neighbor)
    return dequeue

# QUESTION 3: How many extensions does it take to solve the Pokemon problem
#    with DFS (no forward checking) if you do domain reduction before solving it?

ANSWER_3 = 6


def solve_constraint_propagate_reduced_domains(problem):
    """
    Solves the problem using depth-first search with forward checking and
    propagation through all reduced domains.  Same return type as
    solve_constraint_dfs.
    """
    extensions = 0
    agenda = [problem]
    while agenda:
        first_problem = agenda.pop()
        # print(first_problem)
        extensions = extensions + 1
        if has_empty_domains(first_problem):
            continue
        else:
            if not check_all_constraints(first_problem):
                continue
                # print(unassigned_variables)
            else:
                # forward_check(first_problem, next_unassigned_variable)

                unassigned_variables = first_problem.unassigned_vars
                # print(unassigned_variables)
                if unassigned_variables == []:
                    solution = first_problem.assignments
                    return (solution, extensions)
                    # break #solution found
                else:
                    next_unassigned_variable = first_problem.pop_next_unassigned_var()
                    # print(next_unassigned_variable)
                    values_in_domain = first_problem.get_domain(next_unassigned_variable)
                    new_problem_lists = []
                    for val in values_in_domain:
                        new_problem = first_problem.copy().set_assignment(next_unassigned_variable, val)
                        #forward_check(new_problem, next_unassigned_variable)
                        domain_reduction(new_problem,[next_unassigned_variable])
                        new_problem_lists.append(new_problem)
                    # print(agenda)
                    new_problem_lists.reverse()
                    agenda = agenda + new_problem_lists
                    # print("HHHHHHHHH")
                    # print(agenda)
    return (None, extensions)

# QUESTION 4: How many extensions does it take to solve the Pokemon problem
#    with forward checking and propagation through reduced domains?

ANSWER_4 = solve_constraint_propagate_reduced_domains(get_pokemon_problem())[1]


#### Part 5A: Generic Domain Reduction #########################################

def propagate(enqueue_condition_fn, csp, queue=None) :
    """
    Uses constraints to reduce domains, modifying the original csp.
    Uses enqueue_condition_fn to determine whether to enqueue a variable whose
    domain has been reduced. Same return type as domain_reduction.
    """
    dequeue = []
    if queue is None:
        queue = csp.get_all_variables()
    while queue:
        var = queue.pop(0)
        dequeue.append(var)
        reduced_neighbors = eliminate_from_neighbors(csp, var)
        # print(reduced_neighbors)
        if reduced_neighbors is None:
            return None
        for neighbor in reduced_neighbors:
            if (neighbor not in queue) and (enqueue_condition_fn(csp,neighbor)):
                queue.append(neighbor)
    return dequeue

def condition_domain_reduction(csp, var) :
    """Returns True if var should be enqueued under the all-reduced-domains
    condition, otherwise False"""
    return True

def condition_singleton(csp, var) :
    """Returns True if var should be enqueued under the singleton-domains
    condition, otherwise False"""
    if len(csp.get_domain(var)) == 1:
        return True
    return False

def condition_forward_checking(csp, var) :
    """Returns True if var should be enqueued under the forward-checking
    condition, otherwise False"""
    return False

#### Part 5B: Generic Constraint Solver ########################################

def solve_constraint_generic(problem, enqueue_condition=None) :
    """
    Solves the problem, calling propagate with the specified enqueue
    condition (a function). If enqueue_condition is None, uses DFS only.
    Same return type as solve_constraint_dfs.
    """
    extensions = 0
    agenda = [problem]
    while agenda:
        first_problem = agenda.pop()
        # print(first_problem)
        extensions = extensions + 1
        if has_empty_domains(first_problem):
            continue
        else:
            if not check_all_constraints(first_problem):
                continue
                # print(unassigned_variables)
            else:
                # forward_check(first_problem, next_unassigned_variable)

                unassigned_variables = first_problem.unassigned_vars
                if unassigned_variables == []:
                    solution = first_problem.assignments
                    return (solution, extensions)
                else:
                    next_unassigned_variable = first_problem.pop_next_unassigned_var()
                    values_in_domain = first_problem.get_domain(next_unassigned_variable)
                    new_problem_lists = []
                    for val in values_in_domain:
                        new_problem = first_problem.copy().set_assignment(next_unassigned_variable, val)
                        if enqueue_condition is not None:
                            propagate(enqueue_condition, new_problem, [next_unassigned_variable])
                        new_problem_lists.append(new_problem)
                    new_problem_lists.reverse()
                    agenda = agenda + new_problem_lists
    return (None, extensions)


# QUESTION 5: How many extensions does it take to solve the Pokemon problem
#    with forward checking and propagation through singleton domains? (Don't
#    use domain reduction before solving it.)

ANSWER_5 = solve_constraint_generic(get_pokemon_problem(),condition_singleton)[1]


#### Part 6: Defining Custom Constraints #######################################

def constraint_adjacent(m, n) :
    """Returns True if m and n are adjacent, otherwise False.
    Assume m and n are ints."""
    if m-n == 1 or n-m == 1:
        return True
    else:
        return False


def constraint_not_adjacent(m, n) :
    """Returns True if m and n are NOT adjacent, otherwise False.
    Assume m and n are ints."""
    if m-n == 1 or n-m == 1:
        return False
    else:
        return True

def all_different(variables) :
    """Returns a list of constraints, with one difference constraint between
    each pair of variables."""
    result = []
    for i in range(len(variables)):
        for j in range(len(variables)):
            if i < j:

                result.append(Constraint(variables[i], variables[j], constraint_different))

    return result


#### SURVEY ####################################################################

NAME = "XK"
COLLABORATORS = ""
HOW_MANY_HOURS_THIS_LAB_TOOK = 12
WHAT_I_FOUND_INTERESTING = ""
WHAT_I_FOUND_BORING = ""
SUGGESTIONS = ""
