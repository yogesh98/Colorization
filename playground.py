# This file is just for messing around with some to test.

from image_munipulation_functions import *
from PIL import Image


if __name__ == "__main__":
    im = Image.open("Pictures/black.jpg")
    im_as_array = image_to_2d_array(im)

    for row in im_as_array:
        print(row)