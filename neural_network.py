from PIL import Image
import numpy as np
import random
from neural_network_helper import *


class Perceptron:

    def __init__(self, bias):

        self.bias = bias
        self.weights = []


    def add_connection(self, weight, perceptron):
        self.weights.append(weight)
        self.connections.append(perceptron)


if __name__ == "__main__":
    # path = input("Enter Path for the Picture\n")
    path = "Pictures/cat_52x52.jpg"
    try:
        im = Image.open(path)
    except:
        print("File not found")
        quit()

    print("\rTraining Data Created...")
    training_data = create_training_data(im)
    im = training_data[0]
    training_data = training_data[1]

    # im.show()

    print("\rCreating Network")
    weight_matrix = []

    num_layers = 1
    nodes_in_layer = 7
    num_inputs = 9
    biases_matrix = []

    for k in range(num_layers):
        biases_matrix.append([])
        weight_matrix.append([])
        for i in range(nodes_in_layer):
            biases_matrix[k].append(random.uniform(0, 256))
            weight_matrix[k].append([])
            for j in range(num_inputs):
                weight_matrix[k][i].append(random.random())


    nodes_in_output = 3
    biases_matrix.append([])
    weight_matrix.append([])
    for i in range(nodes_in_output):
        biases_matrix[-1].append(random.uniform(0, 256))
        weight_matrix[-1].append([])
        for j in range(nodes_in_layer):
            weight_matrix[-1][-1].append(random.random())

    # biases_matrix = np.array(biases_matrix)
    # weight_matrix = np.array(weight_matrix)


    for answer_inputs in training_data:
        answer = answer_inputs[0]
        inputs = answer_inputs[1]
        input_new = []

        for layer_num in range(len(biases_matrix)):
            layer = biases_matrix[layer_num]
            weights = weight_matrix[layer_num]
            input_new = []
            for weight, bias in zip(weights,layer):
                dp = np.dot(inputs, weight)
                input_new.append(dp)
                print(dp)

            print("\n\n\n\n\n\n")
            inputs = input_new

        output = inputs

        for i in range(3):
            if output[i] < 0:
                output[i] = 0
            elif output[i] > 255:
                output[i] = 255


        error = euclidean_distance(answer, output)

        #TODO back propagation

