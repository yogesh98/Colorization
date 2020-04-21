# This file is just for messing around with some to test.

from image_munipulation_functions import *
from PIL import Image


if __name__ == "__main__":
    # im = Image.open("Pictures/cat_52x52.jpg")
    im = Image.open("Pictures/car1_original_500x333.jpeg")
    im_as_array = image_to_2d_array(im)
    colors = k_means_clustering_on_img(8, im_as_array, 5)

    print(colors)

    for y in range(len(im_as_array) - 1):
        row = im_as_array[y]
        for x in range(len(row) - 1):
            current = row[x]
            ncolor = get_closest(colors, current)
            im_as_array[y][x] = ncolor
            im.putpixel((x, y), ncolor)
    im.show('new.jpg')
