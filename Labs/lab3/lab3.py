 # MIT 6.034 Lab 3: Games
# Written by 6.034 staff

from game_api import *
from boards import *
from toytree import GAME1

INF = float('inf')

# Please see wiki lab page for full description of functions and API.

#### Part 1: Utility Functions #################################################

def is_game_over_connectfour(board):
    """Returns True if game is over, otherwise False."""
    flag = False
    for i in board.get_all_chains(current_player=None):
        if len(i) >= 4: # equal or longer than 4!
            flag = True
    if board.count_pieces(current_player=None) == 42:
        return True
    else:
        return flag
    # return ((board.count_pieces(current_player=None) == 42) or flag)

def next_boards_connectfour(board):
    """Returns a list of ConnectFourBoard objects that could result from the
    next move, or an empty list if no moves can be made."""

    next_moves = []
    for col in range(0,7):
        if not (board.is_column_full(col)):
            next_moves.append(board.add_piece(col))

    if is_game_over_connectfour(board):
        return []
    else:
        return next_moves

def endgame_score_connectfour(board, is_current_player_maximizer):
    """Given an endgame board, returns 1000 if the maximizer has won,
    -1000 if the minimizer has won, or 0 in case of a tie."""
    if is_game_over_connectfour(board):
        for i in board.get_all_chains(current_player=False):
            if len(i) >= 4:
                if (is_current_player_maximizer):
                    return -1000
                else:
                    return 1000
        else:
            return 0


def endgame_score_connectfour_faster(board, is_current_player_maximizer):
    """Given an endgame board, returns an endgame score with abs(score) >= 1000,
    returning larger absolute scores for winning sooner."""
    win_time = board.count_pieces()
    if is_game_over_connectfour(board):
        for i in board.get_all_chains(current_player=False):
            if len(i) >= 4:
                if (is_current_player_maximizer):
                    return -1000 - 42 + win_time
                else:
                    return 1000 + 42 - win_time
        else:
            return 0


def heuristic_connectfour(board, is_current_player_maximizer):
    """Given a non-endgame board, returns a heuristic score with
    abs(score) < 1000, where higher numbers indicate that the board is better
    for the maximizer."""

    chains1 = board.get_all_chains(current_player=True)
    chains2 = board.get_all_chains(current_player=False)
    score1 = 0
    score2 = 0
    for i in chains1:
        if len(i) == 3:
            score1 = score1 + 10
        elif len(i) == 2:
            score1 =score1 + 3
        elif len(i) == 1:
            score1 =score1 + 1

    for i in chains2:
        if len(i) == 3:
            score2 = score2 + 10
        elif len(i) == 2:
            score2 =score2 + 3
        elif len(i) == 1:
            score1 = score1 + 1

    if score1 > score2:
        score = (1 - score2/score1) * 500
        if is_current_player_maximizer:
            return score
        else:
            return -score
    elif score2 > score1:
        score = (1 - score1/score2) * 500
        if is_current_player_maximizer:
            return -score
        else:
            return score
    else:
        return 0


# Now we can create AbstractGameState objects for Connect Four, using some of
# the functions you implemented above.  You can use the following examples to
# test your dfs and minimax implementations in Part 2.

# This AbstractGameState represents a new ConnectFourBoard, before the game has started:
state_starting_connectfour = AbstractGameState(snapshot = ConnectFourBoard(),
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "NEARLY_OVER" from boards.py:


state_NEARLY_OVER = AbstractGameState(snapshot = NEARLY_OVER,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "BOARD_UHOH" from boards.py:
state_UHOH = AbstractGameState(snapshot = BOARD_UHOH,
                                 is_game_over_fn = is_game_over_connectfour,
                                 generate_next_states_fn = next_boards_connectfour,
                                 endgame_score_fn = endgame_score_connectfour_faster)


#### Part 2: Searching a Game Tree #############################################

# Note: Functions in Part 2 use the AbstractGameState API, not ConnectFourBoard.

def dfs_maximizing(state):
    """Performs depth-first search to find path with highest endgame score.
    Returns a tuple containing:
     0. the best path (a list of AbstractGameState objects),
     1. the score of the leaf node (a number), and
     2. the number of static evaluations performed (a number)"""
    state_chains = [[state]]
    leaf_chains = []
    # print(agenda[-1][-1].is_game_over())
    while state_chains:#agenda exists
        current_chain = state_chains.pop()
        current_state = current_chain[-1]
        if not current_state.is_game_over() == True:
            new_states = current_state.generate_next_states()
            # print(new_states)
            new_paths = []
            for i in new_states:
                # print(new_paths)
                # current_state.append(i)
                new_paths.append(current_chain + [i])
                # print(new_paths)
            new_paths.reverse()
            for i in new_paths:
                state_chains.append(i)

        else:
            leaf_chains.append(current_chain)

    max_score = max(path[-1].get_endgame_score(is_current_player_maximizer=True) for path in leaf_chains)
    num = len(leaf_chains)
    for path in leaf_chains:
        if path[-1].get_endgame_score(is_current_player_maximizer=True) == max_score:
            break # get the first matching max path

    result = (path, max_score, num)

    return result

    # raise NotImplementedError
    #Uncomment the line below to try your dfs_maximizing on an
    # AbstractGameState representing the games tree "GAME1" from toytree.py:
    # pretty_print_dfs_type(dfs_maximizing(GAME1))

def minimax_endgame_search(state, maximize=True):
    """Performs minimax search, searching all leaf nodes and statically
    evaluating all endgame scores.  Same return type as dfs_maximizing."""
    results = []
    num = 0
    if state.is_game_over():
        return ([state], state.get_endgame_score(maximize), 1)

    for next_states in state.generate_next_states():#recursion
        results.append(minimax_endgame_search(next_states, not maximize))
    for result in results:
        num += result[2]  # count times reaching the leaf node
    sorting_key = lambda x: x[1]
    if maximize:
        results = max(results, key=sorting_key)
    if not maximize:
        results = min(results, key=sorting_key)
    minimax_path = [state] + results[0]
    return (minimax_path , results[1], num)

# Uncomment the line below to try your minimax_endgame_search on an
# AbstractGameState representing the ConnectFourBoard "NEARLY_OVER" from boards.py:

# pretty_print_dfs_type(minimax_endgame_search(state_NEARLY_OVER))


def minimax_search(state, heuristic_fn=always_zero, depth_limit=INF, maximize=True) :
    """Performs standard minimax search. Same return type as dfs_maximizing."""
    results =[]
    num = 0
    if state.is_game_over():
        return ([state],state.get_endgame_score(maximize),1)
    elif depth_limit == 0:
        return ([state],heuristic_fn(state.get_snapshot(),maximize),1) #must use the snapshot to get the current type of game
    new_depth_limit = depth_limit - 1
    for next_states in state.generate_next_states():  # recursion
        results.append(minimax_search(next_states, heuristic_fn,new_depth_limit,not maximize))
    for result in results:
        num += result[2]  # count times reaching the leaf node
    sorting_key = lambda x: x[1]
    if maximize:
        results = max(results, key=sorting_key)
    if not maximize:
        results = min(results, key=sorting_key)
    minimax_path = [state] + results[0]
    return (minimax_path, results[1], num)

# Uncomment the line below to try minimax_search with "BOARD_UHOH" and
# depth_limit=1. Try increasing the value of depth_limit to see what happens:

# pretty_print_dfs_type(minimax_search(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=1))

def minimax_search_alphabeta(state, alpha=-INF, beta=INF, heuristic_fn=always_zero,
                             depth_limit=INF, maximize=True) :
    """"Performs minimax with alpha-beta pruning. Same return type
    as dfs_maximizing."""
    if state.is_game_over():
        return ([state], state.get_endgame_score(maximize), 1)
    else:
        if depth_limit == 0:
            return ([state], heuristic_fn(state.get_snapshot(), maximize), 1)

        else:
            if maximize:
                best_value = alpha
                next_state = state.generate_next_states()
                num = 0 #compute the times of static evaluatrion
                best_result = minimax_search_alphabeta(next_state[0], best_value, beta, heuristic_fn, depth_limit - 1, False)
                #assume the first next_state is the best, then replace if find better ones; faster than sorting
                for i in next_state:
                    result = minimax_search_alphabeta(i, best_value, beta, heuristic_fn, depth_limit - 1, False)
                    num += result[2]
                    if result[1] > best_value:
                        best_result = result
                        best_value = result[1]
                    if beta <= best_value:
                        break


                return ([state] + best_result[0], best_value, num)
            else:
                best_value = beta
                next_state = state.generate_next_states()
                num = 0
                best_result = minimax_search_alphabeta(next_state[0], alpha, best_value, heuristic_fn, depth_limit - 1, True)
                for i in next_state:
                    result = minimax_search_alphabeta(i, alpha, best_value, heuristic_fn, depth_limit - 1, True)
                    num += result[2]
                    if result[1] < best_value:
                        best_result = result
                        best_value = result[1]
                    if alpha >= best_value:
                        break
                return ([state] + best_result[0], best_value, num)


# Uncomment the line below to try minimax_search_alphabeta with "BOARD_UHOH" and
# depth_limit=4. Compare with the number of evaluations from minimax_search for
# different values of depth_limit.

# pretty_print_dfs_type(minimax_search_alphabeta(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4))


def progressive_deepening(state, heuristic_fn=always_zero, depth_limit=INF,
                          maximize=True) :
    """Runs minimax with alpha-beta pruning. At each level, updates anytime_value
    with the tuple returned from minimax_search_alphabeta. Returns anytime_value."""
    anytime_value = AnytimeValue()
    for i in range(1, depth_limit + 1):
        anytime_value.set_value(minimax_search_alphabeta(state, -INF, INF, heuristic_fn, i, maximize))
    return anytime_value
    raise NotImplementedError

# Uncomment the line below to try progressive_deepening with "BOARD_UHOH" and
# depth_limit=4. Compare the total number of evaluations with the number of
# evaluations from minimax_search or minimax_search_alphabeta.

# progressive_deepening(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4).pretty_print()


# Progressive deepening is NOT optional. However, you may find that 
#  the tests for progressive deepening take a long time. If you would
#  like to temporarily bypass them, set this variable False. You will,
#  of course, need to set this back to True to pass all of the local
#  and online tests.
TEST_PROGRESSIVE_DEEPENING = True
if not TEST_PROGRESSIVE_DEEPENING:
    def not_implemented(*args): raise NotImplementedError
    progressive_deepening = not_implemented


#### Part 3: Multiple Choice ###################################################

ANSWER_1 = '4'

ANSWER_2 = '1'

ANSWER_3 = '4'

ANSWER_4 = '5'


#### SURVEY ###################################################

NAME = "XK"
COLLABORATORS = ""
HOW_MANY_HOURS_THIS_LAB_TOOK = "16"
WHAT_I_FOUND_INTERESTING = ""
WHAT_I_FOUND_BORING = ""
SUGGESTIONS = ""

