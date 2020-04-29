from PIL import Image
import colorsys
from neural_network_helper import *
from neural_network import neural_network

if __name__ == '__main__':

    # path = input("Enter Path for the Picture\n")
    path = "Pictures/red.jpg"
    try:
        im = Image.open(path)
    except:
        print("File not found")
        quit()

    print("\rTraining Data Created...")
    im, training_data = create_training_data(im)
    # im = training_data[0]
    # training_data = training_data[1]
    training_data = list(zip(*training_data))
    RGB = list(zip(*training_data[0]))
    # R = RGB[0]
    # G = RGB[1]
    # B = RGB[2]
    b_w = training_data[1]

    nn = neural_network(9, 5, [9, 10, 9, 137, 1], 3, [None])
    pass

