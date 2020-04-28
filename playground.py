# This file is just for messing around with some to test.

from image_munipulation_functions import *
from PIL import Image


if __name__ == "__main__":
    # im = Image.open("Pictures/Most_similar_test.jpg")
    im = Image.open("Pictures/cat_52x52.jpg")
    im_as_array = image_to_2d_array(im)
    clustered_colors = k_means_clustering_on_img(5, im_as_array, 5)

    # for y in range(len(im_as_array)):
    #     row = im_as_array[y]
    #     for x in range(len(row)):
    #         current = row[x]
    #         ncolor = get_closest(clustered_colors, current)
    #         im.putpixel((x, y), ncolor)


    im.show()

