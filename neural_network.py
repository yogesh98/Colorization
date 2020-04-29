from copy import deepcopy

from PIL import Image
import numpy as np
import random
from neural_network_helper import *

class neural_network:
    #To initialize neural network
    # num_inputs is number of inputs
    # num_layers is the number of hidden layers
    # nodes_in_layer is a list of ints that tell the neural_network object how many nodes are in each layer each index correspondes with that layer
    # nodes_in_output specifies how many output nodes there are
    # activation_function_by_layer is a list of activation functions by layer, each index corresponds to a layer
    # and epoch is epoch lol

    def __init__(self, num_inputs, num_layers, nodes_in_layers, nodes_in_output, activation_function_by_layer, epoch):
        self.num_inputs = num_inputs
        self.num_layers = num_layers
        self.nodes_in_layers = nodes_in_layers
        self.activation_function = activation_function_by_layer
        self.epoch = epoch
        self.weight_matrix = []
        self.biases_matrix = []
        num_inputs = self.num_inputs
        for k in range(num_layers):
            self.biases_matrix.append([])
            self.weight_matrix.append([])
            nodes_in_this_layer = nodes_in_layers[k]
            for i in range(nodes_in_this_layer):
                self.biases_matrix[k].append(random.uniform(0, 256))
                self.weight_matrix[k].append([])
                for j in range(num_inputs):
                    self.weight_matrix[k][i].append(random.random())
            num_inputs = nodes_in_this_layer

        self.biases_matrix.append([])
        self.weight_matrix.append([])
        for i in range(nodes_in_output):
            self.biases_matrix[-1].append(random.uniform(0, 256))
            self.weight_matrix[-1].append([])
            nodes_in_last = nodes_in_layers[-1]
            for j in range(nodes_in_last):
                self.weight_matrix[-1][-1].append(random.random())

    #X_train is a list of training data. The elements of x_train should be lists. Each list contains the inputs specified for the neural network
    #Y_train is the list of actual answers. Each index in y_train corresponds to that same index in X_train
    def train(self,x_train, y_train):
        if self.num_input != len(x_train[0]):
            print("Input doesnt match network input")
            exit()

        for epoch in range(self.epoch):
            for i in range(len(x_train)):
                x = x_train[i]
                answer = y_train[i]

                #Forward Propagation
                out = self.forward_propagation(x)

                #Backward Propagation


    # Given a list of inputs x, this function will return the answer of the neural network
    def forward_propagation(self, x):
        for layer_num in range(len(self.biases_matrix)):
                layer = self.biases_matrix[layer_num]
                weights = self.weight_matrix[layer_num]
                input_new = []
                for weight, bias in zip(weights, layer):
                    dp = np.dot(inputs, weight)
                    input_new.append(dp)
                    print(dp)

                inputs = input_new

        output = activation_function(inputs)
        return output

    # this function is to tune the weights and biases of the neural network
    def backward_probagation:
        #Have to use gradient descent
        pass


# if __name__ == "__main__":
#     # path = input("Enter Path for the Picture\n")
#     path = "Pictures/cat_52x52.jpg"
#     try:
#         im = Image.open(path)
#     except:
#         print("File not found")
#         quit()
#
#     print("\rTraining Data Created...")
#     training_data = create_training_data(im)
#     im = training_data[0]
#     training_data = training_data[1]
#
#     # im.show()
#
#     print("\rCreating Network")
#     weight_matrix = []
#
#     num_layers = 1
#     nodes_in_layer = 9
#     num_inputs = 9
#     biases_matrix = []
#
#     for k in range(num_layers):
#         biases_matrix.append([])
#         weight_matrix.append([])
#         for i in range(nodes_in_layer):
#             biases_matrix[k].append(random.uniform(0, 256))
#             weight_matrix[k].append([])
#             for j in range(num_inputs):
#                 weight_matrix[k][i].append(random.random())
#
#
#     nodes_in_output = 3
#     biases_matrix.append([])
#     weight_matrix.append([])
#     for i in range(nodes_in_output):
#         biases_matrix[-1].append(random.uniform(0, 256))
#         weight_matrix[-1].append([])
#         for j in range(nodes_in_layer):
#             weight_matrix[-1][-1].append(random.random())
#
#     # biases_matrix = np.array(biases_matrix)
#     # weight_matrix = np.array(weight_matrix)
#
#
#     for answer_inputs in training_data:
#         answer = answer_inputs[0]
#         inputs = answer_inputs[1]
#         input_new = []
#
#         for layer_num in range(len(biases_matrix)):
#             layer = biases_matrix[layer_num]
#             weights = weight_matrix[layer_num]
#             input_new = []
#             for weight, bias in zip(weights,layer):
#                 dp = np.dot(inputs, weight)
#                 input_new.append(dp)
#                 print(dp)
#
#             print("\n\n\n\n\n\n")
#             inputs = input_new
#
#
#         output = activation_function(inputs)
#
#
#         error = euclidean_distance(answer, output)
#
#         #TODO back propagation

