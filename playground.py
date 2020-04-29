# This file is just for messing around with some to test.

from basic_agent_helper_functions import *
from PIL import Image


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

if __name__ == "__main__":
    im = Image.open("Pictures/Beach1_original_500x357.jpg")
    # im = Image.open("Pictures/cat_52x52.jpg")
    im_as_array = image_to_2d_array(im)
    colors = get_html_5_colors()

    print(len(colors))

    for y in range(len(im_as_array)):
        row = im_as_array[y]
        for x in range(len(row)):
            print("\rWorking on (" + str(x) + ", " + str(y) + ")", end='')
            current = row[x]
            ncolor = get_closest(colors, current)
            im.putpixel((x, y), ncolor)


    im.show()

