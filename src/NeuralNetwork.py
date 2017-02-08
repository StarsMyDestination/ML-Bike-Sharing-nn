import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def MSE(y, Y):
    return np.mean((y - Y)**2)


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes**-0.5,
                                                        (self.hidden_nodes, self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes**-0.5,
                                                         (self.output_nodes, self.hidden_nodes))
        self.lr = learning_rate

        # Set this to your implemented sigmoid function ####
        # Activation function is the sigmoid function
        self.activation_function = sigmoid

    def train(self, inputs_list, targets_list):
        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2)  # inputs is a m*n matrix, m-batch_size; n-feature_number
        targets = np.array(targets_list, ndmin=2)  # targets is a m*1 matrix, m-batch_size; 1-output unit
        n_records, n_features = inputs.shape

        # Implement the forward pass here ####
        # Hidden layer
        hidden_inputs = np.matmul(inputs, self.weights_input_to_hidden.T)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

        # TODO: Output layer
        final_inputs = np.matmul(hidden_outputs, self.weights_hidden_to_output.T)  # signals into final output layer
        final_outputs = final_inputs  # signals from final output layer

        # Implement the backward pass here ####
        # Output error
        output_errors = (targets - final_outputs) * 1.0  # Output layer error is the difference between desired target and actual output.
        output_grad = np.zeros((self.output_nodes, self.hidden_nodes))
        for i in range(n_records):
            output_grad += np.matmul(output_errors[i, :].T, hidden_outputs[i, :])
        output_grad = output_grad / float(n_records)
        # Backpropagated error
        hidden_errors = np.multiply(np.matmul(output_errors, self.weights_hidden_to_output), np.multiply(
            hidden_outputs, (1 - hidden_outputs)))  # errors propagated to the hidden layer
        hidden_grad = 0
        for i in range(n_records):
            hidden_grad += np.matmul(hidden_errors[i, :].T, inputs[i, :])
        hidden_grad = hidden_grad / float(n_records)  # hidden layer gradients

        # Update the weights
        self.weights_hidden_to_output += self.lr * output_grad  # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * hidden_grad  # update input-to-hidden weights with gradient descent step

    def run(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T

        # Implement the forward pass here ####
        # TODO: Hidden layer
        hidden_inputs = np.matmul(inputs, self.weights_input_to_hidden.T)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

        # TODO: Output layer
        final_inputs = np.matmul(hidden_outputs, self.weights_hidden_to_output.T)  # signals into final output layer
        final_outputs = final_inputs  # signals from final output layer

        return final_outputs
