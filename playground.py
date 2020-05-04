# This file is just for messing around with some to test.

from PIL import Image
from neural_network_helper import *
from sklearn.model_selection import train_test_split
from mv_linear_regression import LinearRegression
import matplotlib.pyplot as plt


if __name__ == "__main__":
    path = input("Enter Path for the Picture\n")
    # path = "Pictures/red.jpg"
    # path = "Pictures/cat_52x52.jpg"
    try:
        im = Image.open(path)
    except:
        print("File not found")
        quit()

    im_as_array = image_to_2d_array(im)
    im_gray = im.convert("L")
    im_gray_as_array = image_to_2d_array(im_gray)

    x = []
    y = []

    for row in range(2, len(im_gray_as_array)-2):
        for col in range(2, len(im_gray_as_array[0]) -2):
            _sum = 0
            for y2 in range(row-2,row+3):
                for x2 in range(col-2,col+3):
                    _sum += im_gray_as_array[y2][x2]
            x.append(_sum)
            y.append(im_as_array[row][col])

    R, G, B = zip(*y)

    colors = get_html_5_colors()
    for val in range(len(y)):
        y[val] = get_closest_index(colors, y[val])

    plt.scatter(x,y)
    plt.show()
    plt.clf()
    plt.scatter(x, R)
    plt.show()
    plt.scatter(x, G)
    plt.show()
    plt.scatter(x, B)
    plt.show()