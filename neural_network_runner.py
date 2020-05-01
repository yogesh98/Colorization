from PIL import Image
import colorsys
from neural_network_helper import *
from neural_network import neural_network
from itertools import permutations

if __name__ == '__main__':

    # path = input("Enter Path for the Picture\n")
    # # path = "Pictures/red.jpg"
    # try:
    #     im = Image.open(path)
    # except:
    #     print("File not found")
    #     quit()
    #
    # print("\rTraining Data Created...")
    # im, training_data = create_training_data(im)
    # # im = training_data[0]
    # # training_data = training_data[1]
    # training_data = list(zip(*training_data))
    # RGB = list(zip(*training_data[0]))
    # # R = RGB[0]
    # # G = RGB[1]
    # # B = RGB[2]
    # b_w = training_data[1]
    # fake_train = []
    # fake_answers = []
    # for i in range(1, 10):
    #     fake_train.append([i])
    #     fake_answers.append([i])
    # nn = neural_network(1,1,[1],1,[sigmoid,relu],2000,100,1)
    # nn.train(fake_train, fake_answers)
    # print(nn.run([3]))
    # print(nn.run([11]))

    # training_data = list(permutations([1,-1,2,-2]))
    #
    # for i in range(len(training_data)):
    #     training_data[i] = list(training_data[i])
    #     for num in range(len(training_data[i])):
    #         if training_data[i][num] == -2:
    #             training_data[i][num] = -1
    #         elif training_data[i][num] == 2:
    #             training_data[i][num] = 1

    training_data = [[1, -1, 1, -1], [1, -1, -1, 1], [1, 1, -1, -1], [1, 1, -1, -1], [1, -1, -1, 1], [1, -1, 1, -1], [-1, 1, 1, -1], [-1, 1, -1, 1], [-1, 1, 1, -1], [-1, 1, -1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [1, 1, -1, -1], [1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1], [1, -1, 1, -1], [1, -1, -1, 1], [-1, 1, -1, 1], [-1, 1, 1, -1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, 1, 1, -1], [-1, 1, -1, 1]]
    answers = [[0], [0], [1], [1], [0], [0], [0], [0], [0], [0], [1], [1], [1], [1], [0], [0], [0], [0], [0], [0], [1], [1], [0], [0]]


    nn = neural_network(4, 2, [4, 2], 1, [na, na, na], 1, 1, 1)
    nn.train(training_data,answers)


    print(answers)
