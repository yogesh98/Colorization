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

    def __init__(self, num_inputs, num_layers, nodes_in_layers, nodes_in_output, activation_function_by_layer, epoch, learning_rate, momentum):
        random.seed(49)
        self.num_inputs = num_inputs
        self.num_layers = num_layers
        self.nodes_in_layers = nodes_in_layers
        self.nodes_in_output = nodes_in_output
        self.activation_function = activation_function_by_layer
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_matrices = []
        self.biases_matrices = []
        num_inputs = self.num_inputs
        for k in range(num_layers):
            self.biases_matrices.append([])
            self.weight_matrices.append([])
            nodes_in_this_layer = nodes_in_layers[k]
            for i in range(nodes_in_this_layer):
                # self.biases_matrices[k].append(-1.0)
                # self.biases_matrices[k].append(random.uniform(0, 256))
                self.biases_matrices[k].append(random.random())
                self.weight_matrices[k].append([])
                for j in range(num_inputs):
                    # self.weight_matrices[k][i].append((k + 1) * (i + 1))
                    # self.weight_matrices[k][i].append(random.randint(1,10) * 1.0)
                    self.weight_matrices[k][i].append(random.random())
            num_inputs = nodes_in_this_layer

        self.biases_matrices.append([])
        self.weight_matrices.append([])
        for i in range(nodes_in_output):
            # self.biases_matrices[-1].append(0.0)
            # self.biases_matrices[-1].append(random.uniform(0, 256))
            self.biases_matrices[-1].append(random.random())
            self.weight_matrices[-1].append([])
            nodes_in_last = nodes_in_layers[-1]
            for j in range(nodes_in_last):
                # choose = int(input("Choose for output layer"))
                # self.weight_matrices[-1][-1].append(choose)
                # self.weight_matrices[-1][-1].append(random.randint(1,10) * 1.0)
                self.weight_matrices[-1][-1].append(random.random())

        # self.weight_matrices = [[[-1, -1, 0, 0], [1, 1, 0, 0], [0, 0, -1, -1], [0, 0, 1, 1]],
        #                         [[0, 1, 1, 0], [1, 0, 0, 1]],
        #                         [[1, 1]]]
        # print(self.weight_matrices)

        # self.weight_matrices = [[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        #                         [[1, 1, 1, 1], [1, 1, 1, 1]],
        #                         [[1, 1]]]


        for i in range(len(self.biases_matrices)):
            self.biases_matrices[i] = np.array(self.biases_matrices[i])
            self.weight_matrices[i] = np.array(self.weight_matrices[i])
        self.biases_matrices = np.array(self.biases_matrices)
        self.weight_matrices = np.array(self.weight_matrices)

    #X_train is a list of training data. The elements of x_train should be lists. Each list contains the inputs specified for the neural network
    #Y_train is the list of actual answers. Each index in y_train corresponds to that same index in X_train
    def train(self, x_train, y_train):
        if self.num_inputs != len(x_train[0]):
            print("Input doesnt match network input")
            exit()

        for epoch in range(self.epoch):
            cost = 0
            for i in range(len(x_train)):
                x = x_train[i]
                answer = y_train[i]

                #Forward Propagation
                out, output_by_layer = self.forward_propagation(x)
                e = error(out, answer)
                for val in e:
                    cost += val
                #Backward Propagation
                self.backward_propagation(output_by_layer, e)
            cost = cost ** 2
            print("epoch: %d\tTotalCost: %f" % (epoch, cost))
            # print("\tWeights: " + str(self.weight_matrices))



    # Given a list of inputs x, this function will return the answer of the neural network
    def forward_propagation(self, x):
        inputs = x
        output_by_layer = [inputs]
        for layer_num in range(len(self.biases_matrices)):
                layer = self.biases_matrices[layer_num]
                weights = self.weight_matrices[layer_num]
                input_new = []
                for weight, bias in zip(weights, layer):
                    dp = np.dot(np.array(inputs), weight) + bias
                    input_new.append(dp)
                    # print(dp)
                inputs = list(map(self.activation_function[layer_num], input_new))
                output_by_layer.append(inputs)
        output = inputs
        for i in range(len(output_by_layer)):
            output_by_layer[i] = np.array(output_by_layer[i])
        output_by_layer = np.array(output_by_layer)
        return output, output_by_layer

    # this function is to tune the weights and biases of the neural network
    def backward_propagation(self, output_by_layer, err):
        derivative_list = create_df_list(self.activation_function)
        # Have to use gradient descent
        err = np.array(err)
        err_matrix = [err]
        for i in range(len(self.weight_matrices)-1 , -1, -1):
            transposed_weights = np.transpose(self.weight_matrices[i])
            err = transposed_weights @ err
            err_matrix.insert(0, err)

        D = deepcopy(output_by_layer)
        for i in range(1, len(output_by_layer)):
            D[i] = derivative_list[i-1](output_by_layer[i])


        # for i in range(len(self.weight_matrices)):
        #     temp = err_matrix[i]
        #     temp = temp * np.transpose(D[i])
        #     trans = np.transpose(output_by_layer[i])
        #     # print(type(temp))
        #     temp = temp @ trans
        #
        #     gradient = temp
        #
        #     delta_weights = self.weight_matrices[]

        for layer_num in range(len(self.weight_matrices)-1, -1, -1):
            matrix = self.weight_matrices[layer_num]
            d_matrix = D[layer_num + 1]
            for entry in range(len(matrix)):
                matrix[entry] = matrix[entry] - (d_matrix[entry] * self.learning_rate)
            self.weight_matrices[layer_num] = matrix
            biases = self.biases_matrices[layer_num]
            for entry in range(len(biases)):
                biases[entry] -= d_matrix[entry]
            self.biases_matrices[layer_num] = biases
    def run(self, input):
        return(self.forward_propagation(input)[0])


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
#     weight_matrices = []
#
#     num_layers = 1
#     nodes_in_layer = 9
#     num_inputs = 9
#     biases_matrices = []
#
#     for k in range(num_layers):
#         biases_matrices.append([])
#         weight_matrices.append([])
#         for i in range(nodes_in_layer):
#             biases_matrices[k].append(random.uniform(0, 256))
#             weight_matrices[k].append([])
#             for j in range(num_inputs):
#                 weight_matrices[k][i].append(random.random())
#
#
#     nodes_in_output = 3
#     biases_matrices.append([])
#     weight_matrices.append([])
#     for i in range(nodes_in_output):
#         biases_matrices[-1].append(random.uniform(0, 256))
#         weight_matrices[-1].append([])
#         for j in range(nodes_in_layer):
#             weight_matrices[-1][-1].append(random.random())
#
#     # biases_matrices = np.array(biases_matrices)
#     # weight_matrices = np.array(weight_matrices)
#
#
#     for answer_inputs in training_data:
#         answer = answer_inputs[0]
#         inputs = answer_inputs[1]
#         input_new = []
#
#         for layer_num in range(len(biases_matrices)):
#             layer = biases_matrices[layer_num]
#             weights = weight_matrices[layer_num]
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

