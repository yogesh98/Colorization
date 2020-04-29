import math
from basic_agent_helper_functions import *


def get_html_5_colors():
    colors = []
    with open("HTML_5_colors.txt") as f:
        temp = []
        for line in f:
            line = line.replace("(", "").replace(")", "")
            for token in line.split(", "):
                num = int(token)
                temp.append(num)
            colors.append(tuple(temp))
            temp = []
    return colors


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def create_training_data(im):
    training_data_nn = []
    im_as_array = image_to_2d_array(im)

    im_gray = im.convert("L")
    im_gray_as_array = image_to_2d_array(im_gray)

    im = im.convert("L")
    im = im.convert("RGB")

    colors = get_html_5_colors()

    w = im.size[0]
    h = im.size[1]

    for y in range(h):
        row = im_as_array[y]
        for x in range(int(round(w/2)), w):
            print("\rWorking on (" + str(x) + ", " + str(y) + ")", end='')
            current = row[x]
            ncolor = get_closest(colors, current)
            im.putpixel((x, y), ncolor)

            temp = []
            try:
                for y2 in range(y-1,y+2):
                    for x2 in range(x-1,x+2):
                        temp.append(im_gray_as_array[y2][x2])
                training_data_nn.append((ncolor, temp))
            except IndexError:
                pass

    return im, training_data_nn

