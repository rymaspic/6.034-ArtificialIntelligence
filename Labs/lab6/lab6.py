# MIT 6.034 Lab 6: Neural Nets
# Written by 6.034 Staff

from nn_problems import *
from math import e
INF = float('inf')


#### Part 1: Wiring a Neural Net ###############################################

nn_half = [1]

nn_angle = [2,1]#Q

nn_cross = [2,2,1]

nn_stripe = [3,1]

nn_hexagon = [6,1]

nn_grid = [4,2,1]


#### Part 2: Coding Warmup #####################################################

# Threshold functions
def stairstep(x, threshold=0):
    "Computes stairstep(x) using the given threshold (T)"
    return x >= threshold

def sigmoid(x, steepness=1, midpoint=0):
    "Computes sigmoid(x) using the given steepness (S) and midpoint (M)"
    return 1/(1 + e**(- steepness*(x-midpoint)))

def ReLU(x):
    "Computes the threshold of an input using a rectified linear unit."
    return max(0,x)

# Accuracy function
def accuracy(desired_output, actual_output):
    "Computes accuracy. If output is binary, accuracy ranges from -0.5 to 0."
    return -(1/2) * (desired_output - actual_output)**2

#### Part 3: Forward Propagation ###############################################

def node_value(node, input_values, neuron_outputs):  # PROVIDED BY THE STAFF
    """
    Given 
     * a node (as an input or as a neuron),
     * a dictionary mapping input names to their values, and
     * a dictionary mapping neuron names to their outputs
    returns the output value of the node.
    This function does NOT do any computation; it simply looks up
    values in the provided dictionaries.
    """
    if isinstance(node, str):
        # A string node (either an input or a neuron)
        if node in input_values:
            return input_values[node]
        if node in neuron_outputs:
            return neuron_outputs[node]
        raise KeyError("Node '{}' not found in either the input values or neuron outputs dictionary.".format(node))
    
    if isinstance(node, (int, float)):
        # A constant input, such as -1
        return node
    
    raise TypeError("Node argument is {}; should be either a string or a number.".format(node))

def forward_prop(net, input_values, threshold_fn=stairstep):
    """Given a neural net and dictionary of input values, performs forward
    propagation with the given threshold function to compute binary output.
    This function should not modify the input net.  Returns a tuple containing:
    (1) the final output of the neural net
    (2) a dictionary mapping neurons to their immediate outputs"""

    final_output = 0
    neural_outputs = {}
    #iterate each neuron in the network
    for neuron in net.topological_sort():
        total_output = 0
        for input in net.get_incoming_neighbors(neuron): #input may be a neural output that is not "input node"
            weighted_input = net.get_wires(input, neuron)[0].get_weight() * node_value(input, input_values, neural_outputs)
            total_output = total_output + weighted_input
        output = threshold_fn(total_output)
        neural_outputs.update({neuron:output})

    for neuron in net.topological_sort():
        if net.is_output_neuron(neuron):
            final_output = neural_outputs[neuron]

    return (final_output,neural_outputs)

#### Part 4: Backward Propagation ##############################################

def gradient_ascent_step(func, inputs, step_size):
    """Given an unknown function of three variables and a list of three values
    representing the current inputs into the function, increments each variable
    by +/- step_size or 0, with the goal of maximizing the function output.
    After trying all possible variable assignments, returns a tuple containing:
    (1) the maximum function output found, and
    (2) the list of inputs that yielded the highest function output."""

    options = [+step_size,-step_size,0]

    new_input1 = []
    new_input2 = []
    new_input3 = []
    max_inputs = []

    for option in options:
        new_input1.append(inputs[0] + option)
        new_input2.append(inputs[1] + option)
        new_input3.append(inputs[2] + option)

    max_output = func(new_input1[0],new_input2[0],new_input3[0])

    for input1 in new_input1:
        for input2 in new_input2:
            for input3 in new_input3:
                if func(input1,input2,input3) > max_output:
                    max_output = func(input1,input2,input3)

    for input1 in new_input1:
        for input2 in new_input2:
            for input3 in new_input3:
                if func(input1,input2,input3) == max_output:
                    max_inputs.append([input1, input2, input3])

    return (max_output,max_inputs[0])

def get_back_prop_dependencies(net, wire):
    """Given a wire in a neural network, returns a set of inputs, neurons, and
    Wires whose outputs/values are required to update this wire's weight."""

    if net.is_output_neuron(wire.endNode):
        results = [wire.startNode, wire, wire.endNode]
        #print(wire.endNode)
        return set(results)
    else:
        results = [wire,wire.startNode]
        for next in net.get_outgoing_neighbors(wire.endNode):
            results.extend(get_back_prop_dependencies(net,net.get_wires(wire.endNode,next)[0]))
            #return set(get_back_prop_dependencies(net, net.get_wires(wire.endNode, next)[0]))
        return set(results)
    raise NotImplementedError

def calculate_deltas(net, desired_output, neuron_outputs):
    """Given a neural net and a dictionary of neuron outputs from forward-
    propagation, computes the update coefficient (delta_B) for each
    neuron in the net. Uses the sigmoid function to compute neuron output.
    Returns a dictionary mapping neuron names to update coefficient (the
    delta_B values). """

    deltas = {}

    for neuron in net.topological_sort()[::-1]:
        if net.is_output_neuron(neuron):
            deltas[neuron] = neuron_outputs[neuron] * \
                             (1 - neuron_outputs[neuron]) * (desired_output - neuron_outputs[neuron])

        else:
            total_output = 0
            for neighbor in net.get_outgoing_neighbors(neuron):
                wire = net.get_wires(neuron, neighbor)
                total_output += deltas[neighbor] * wire[0].get_weight()

            deltas[neuron] = neuron_outputs[neuron] * (1 - neuron_outputs[neuron]) * total_output

    return deltas


def update_weights(net, input_values, desired_output, neuron_outputs, r=1):
    """Performs a single step of back-propagation.  Computes delta_B values and
    weight updates for entire neural net, then updates all weights.  Uses the
    sigmoid function to compute neuron output.  Returns the modified neural net,
    with the updated weights."""

    deltas = calculate_deltas(net, desired_output, neuron_outputs)
    for wire in net.get_wires():
        start = wire.startNode
        end = wire.endNode
        weight = wire.get_weight()
        wire.set_weight(weight + r * node_value(start,input_values,neuron_outputs) * deltas[end])
    return net

def back_prop(net, input_values, desired_output, r=1, minimum_accuracy=-0.001):
    """Updates weights until accuracy surpasses minimum_accuracy.  Uses the
    sigmoid function to compute neuron output.  Returns a tuple containing:
    (1) the modified neural net, with trained weights
    (2) the number of iterations (that is, the number of weight updates)"""

    iterations = 0
    current_accuracy = accuracy(desired_output,(forward_prop(net,input_values,sigmoid)[0]))
    while current_accuracy <= minimum_accuracy:
        net = update_weights(net, input_values, desired_output, forward_prop(net,input_values,sigmoid)[1], r)
        current_accuracy = accuracy(desired_output, (forward_prop(net, input_values, sigmoid)[0]))
        iterations = iterations + 1

    return (net, iterations)

#### Part 5: Training a Neural Net #############################################

ANSWER_1 = 17
ANSWER_2 = 14
ANSWER_3 = 2
ANSWER_4 = 80
ANSWER_5 = 60

ANSWER_6 = 1
ANSWER_7 = 'checkerboard' #Q
ANSWER_8 = ['small', 'medium', 'large']
ANSWER_9 = 'B'

ANSWER_10 = 'D' #Q
ANSWER_11 = ['A','C']
ANSWER_12 = ['A','E']


#### SURVEY ####################################################################

NAME = 'XK'
COLLABORATORS = ''
HOW_MANY_HOURS_THIS_LAB_TOOK = 10
WHAT_I_FOUND_INTERESTING = ''
WHAT_I_FOUND_BORING = ''
SUGGESTIONS = ''
