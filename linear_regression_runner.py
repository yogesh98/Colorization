import sys

from PIL import Image
from neural_network_helper import *
from sklearn.model_selection import train_test_split
from mv_linear_regression import LinearRegression

if __name__ == '__main__':

    # path = input("Enter Path for the Picture\n")
    # path = "Pictures/red.jpg"
    path = "Pictures/cat_52x52.jpg"
    try:
        im = Image.open(path)
    except:
        print("File not found")
        quit()

    im, X, y = create_training_data(im)
    print("\rTraining Data Created...")

    im_as_array = image_to_2d_array(im)
    im_gray = im.convert("L")
    im_gray_as_array = image_to_2d_array(im_gray)

    X = np.array(X)
    y = np.array(y)

    mean = X.mean()
    max = X.max()
    min = X.min()

    if max - min != 0:
        X = (X - mean) / (max - min)
    else:
        X = (X - mean)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)

    X_train = X_train.T
    X_test = X_test.T
    y_train = np.array([y_train])
    y_test = np.array([y_test])

    model = LinearRegression(X_train, y_train, X_test, y_test, .1, 5000)

    for row_num in range(1, len(im_as_array) - 1):
        row = im_as_array[row_num]
        for col_num in range(1, int(round((len(row)) / 2))):
            print("\rWorking on (" + str(col_num) + ", " + str(row_num) + ")", end='')
            sys.stdout.flush()
            input = []
            for y2 in range(row_num-1,row_num+2):
                for x2 in range(col_num-1,col_num+2):
                    input.append(im_gray_as_array[y2][x2])

            input = np.array(input)

            print(model.forward_prop(input))



    pass