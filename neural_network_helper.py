import math
import numpy as np
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

# def error(x, y):
#     e = 0
#     for xvalue, yvalue in zip(x, y):
#         e += (xvalue - yvalue) ** 2
#     return e

def error(output, answer):
    return [abs(x - y) for x,y in zip(output, answer)]

def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    if np.isnan(sig):
        return 0
    return sig


#assuming x is already been through the sigmoid function
def dsigmoid(x):
    return x * (1 - x)

def na(x):
    if x > 0:
        return 1
    return 0
    return x

def drelu(x):
    if x > 0:
        return 1
    return 0

def relu(x):
    return max([0, x])


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


def cap_output(output):
    for i in range(3):
        if output[i] < 0:
            output[i] = 0
        elif output[i] > 255:
            output[i] = 255
    return output

def create_df_list(af_list):
    df_list = []
    for f in af_list:
        if f == sigmoid:
            df_list.append(np.vectorize(dsigmoid))
        elif f == relu:
            df_list.append(np.vectorize(drelu))
        else:
            df_list.append(np.vectorize(drelu))
    return df_list
# print(error([1, 2, 3], [1, 45, 3]))

# for i in np.linspace(0,20000,200000):
#     i = -1 * round(i,7)
#     # sigmoid(i)
#     print(str(i) + ": " + str(sigmoid(i)))