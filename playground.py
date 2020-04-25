# This file is just for messing around with some to test.

from image_munipulation_functions import *
from PIL import Image


if __name__ == "__main__":
    im = Image.open("Pictures/Most_similar_test.jpg")
    # im = Image.open("Pictures/cat_52x52.jpg")
    # im_as_array = image_to_2d_array(im)
    # clustered_colors = k_means_clustering_on_img(5, im_as_array, 5)
    # imFirst = left_gray_right_colored(clustered_colors, im, im_as_array)
    # imFirst.show('new.jpg')

    print()
    im_gray = im.convert("L")
    im_gray_as_array = image_to_2d_array(im_gray)
    print(six_similar_on_right_half(im_gray_as_array, 1, 1))


    #print(clustered_colors)

