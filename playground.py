# This file is just for messing around with some to test.

from image_munipulation_functions import *
from PIL import Image


if __name__ == "__main__":
    im = Image.open("Pictures/Beach1_original_500x333.jpeg")
    im_as_array = image_to_2d_array(im)
    print(k_means_clustering_on_img(4, im_as_array, 1))