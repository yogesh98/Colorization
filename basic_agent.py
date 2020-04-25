import sys

from image_munipulation_functions import *
from PIL import Image


if __name__ == "__main__":
    path = input("Enter Path for the Picture\n")

    try:
        im = Image.open(path)
    except:
        print("File not found")
        quit()

    print("Clustering Colors...", end='')
    im_as_array = image_to_2d_array(im)
    clustered_colors = k_means_clustering_on_img(5, im_as_array, 5)

    im_final = left_gray_right_colored(clustered_colors, im, im_as_array)
    im_final_as_array = image_to_2d_array(im_final)

    im_gray = im.convert("L")
    im_gray_as_array = image_to_2d_array(im_gray)

    print("\rthis picture is " + str(len(im_final_as_array[0])) + " x " + str(len(im_final_as_array)))

    for row_num in range(1, len(im_final_as_array) - 1):
        row = im_final_as_array[row_num]
        for col_num in range(1, int(round((len(row)) / 2))):
            print("\rWorking on (" + str(col_num) + ", " + str(row_num) + ")", end='')
            sys.stdout.flush()
            similar_patches = six_similar_on_right_half(im_gray_as_array, col_num, row_num)

            temp = list(zip(*similar_patches))

            similarity_index = temp[0]
            coordinates = temp[1]

            colors_at_coordinates = list(map(lambda c: im_final_as_array[c[1]][c[0]], coordinates))
            similar_patches_w_colors = zip(similarity_index, colors_at_coordinates)


            colors_at_coordinates_as_set = set(colors_at_coordinates)
            colors_w_count = list(map(lambda c: [0,c], colors_at_coordinates_as_set))

            for color in colors_w_count:
                for c in colors_at_coordinates:
                    if color[1] == c:
                        color[0] += 1

            max = -1
            possible_new_color = []
            for color in colors_w_count:
                if color[0] > max:
                    max = color[0]
                    possible_new_color = [color[1]]
                elif color[0] == max:
                    possible_new_color.append(color[1])

            new_color = None
            if len(possible_new_color) > 1:
                most_similar = []
                least = float('inf')
                for color in possible_new_color:
                    for patch in similar_patches_w_colors:
                        if patch[1] in possible_new_color:
                            if patch[0] < least:
                                most_similar = [patch[1]]
                                least = patch[0]
                            elif patch[0] == least:
                                most_similar.append(patch[1])
                new_color = random.choice(most_similar)
            else:
                new_color = possible_new_color[0]

            im_final.putpixel((col_num, row_num), new_color)
    print("\rIf final image did not show up automatically, you can find the image saved as new.jpg")
    sys.stdout.flush()
    im_final.save("new.jpg")
    im_final.show()