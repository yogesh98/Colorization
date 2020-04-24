# This file is just for messing around with some to test.

from image_munipulation_functions import *
from PIL import Image


if __name__ == "__main__":
    im = Image.open("Pictures/cat_52x52.jpg")
    #im = Image.open("Pictures/car1_original_500x333.jpeg")
    im_as_array = image_to_2d_array(im)
    colors = k_means_clustering_on_img(5, im_as_array, 5)
    imFirst = image_first_transform(colors, im, im_as_array)
    imFirst.show('new.jpg')



    #print(colors)

