from PIL import Image, ImageOps
import random
import math
import operator
from PIL import Image

training_data = []
# Takes Image object and returns a 2d list of RGB values as tuples in the lists
def image_to_2d_array(im):
    w = im.size[0]
    h = im.size[1]

    arr = list(im.getdata())

    return [arr[i:i + w] for i in range(0, len(arr), w)]


# takes 2d list of RGB tuples and performs a k means clustering on them
# Returns the k clustered_colors that it found in the clustering
# k is the number of clusters, im is the image as a 2d list of rgb tuples
# and num attempts is how many attempts it should make to find the best clusters
def k_means_clustering_on_img(k, im, num_attempts):
    clustering_attempts = []
    while len(clustering_attempts) < num_attempts:
        # print("Trying: " + str(len(clustering_attempts)))
        # Randomly selecting starting clusters
        centers = []
        while len(centers) < k:
            choice = random.choice(random.choice(im))
            if choice not in centers:
                centers.append(choice)

        # group points in clusters by selecting nearest cluster
        # calculate the mean of each cluster, and recluster using mean values
        # keep doing this until mean values do not change
        # also keep track of distances in another list so its easier to calculate variation later
        centroid_changed = True
        clusters = []
        for i in centers:
            clusters.append([])
        while centroid_changed:
            centroid_changed = False
            for row in im:
                for current in row:
                    closest_centroid_indices = []
                    smallest_distance = float('inf')
                    for i in range(len(centers)):
                        centroid = centers[i]
                        distance = euclidean_distance(current, centroid)

                        if distance < smallest_distance:
                            smallest_distance = distance
                            closest_centroid_indices = [i]
                        elif distance == smallest_distance:
                            closest_centroid_indices.append(i)

                    center_index = random.choice(closest_centroid_indices)
                    clusters[center_index].append(current)
            for i in range(len(centers)):
                center = centers[i]
                cluster = clusters[i]
                mean = mean_point(cluster)

                if mean != center:
                    centers[i] = mean
                    centroid_changed = True
        variances = []
        for i in range(len(centers)):
            center = centers[i]
            cluster = clusters[i]
            distance_vector = get_distance_vector(center, cluster)
            variances.append(variance(distance_vector))
        clustering_attempts.append((centers, sum(variances)))

    min_variance = float('inf')
    min_clusters = []
    for attempt in clustering_attempts:
        cluster = attempt[0]
        var = attempt[1]
        if var < min_variance:
            min_clusters.append(cluster)

    return random.choice(min_clusters)


def left_gray_right_colored(colors, im, im_as_array):
    imGray = im.convert("L")
    imGray = imGray.convert("RGB")
    im_as_arrayNew = image_to_2d_array(imGray)
    for y in range(len(im_as_array)):
        row = im_as_array[y]
        for x in range(int(round((len(row))/2)), (len(row))):
            current = row[x]
            ncolor = get_closest(colors, current)
            im_as_arrayNew[y][x] = ncolor
            imGray.putpixel((x, y), ncolor)
    return imGray


def euclidean_distance(start, end):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(start, end)]))


def mean_point(lst):
    return (int(round(sum([point[0] for point in lst])/len(lst))),
            int(round(sum([point[1] for point in lst])/len(lst))),
            int(round(sum([point[2] for point in lst])/len(lst))))


def variance(lst):
    mean = sum(lst)/len(lst)
    return sum((i - mean) ** 2 for i in lst) / len(lst)


def get_distance_vector(center, cluster):
    return [euclidean_distance(center, x) for x in cluster]


def get_closest(centers, current):
    min_dist = float('inf')
    closest = []
    for i in centers:
        dist = euclidean_distance(i, current)

        if dist < min_dist:
            min_dist = dist
            closest = [i]
        elif dist == min_dist:
            closest.append(i)

    return random.choice(closest)

def initialize_training_data(imGray_as_array):
    num_rows = len(imGray_as_array)
    num_cols = len(imGray_as_array[0])

    for y in range(num_rows):
        training_data.append([])
        for x in range(num_cols):
            value = []
            if x == 0 or x == num_cols - 1 or y == 0 or y == num_rows - 1:
                training_data[y].append(None)
            else:
                for y2 in range(y - 1, y + 2):
                    for x2 in range(x - 1, x + 2):
                        value.append(imGray_as_array[y2][x2])
                training_data[y].append(value)

    # for row in training_data:
    #     print(row)

#Changed to be given already grayscale image to it runs quicker
def six_similar_on_right_half(imGray_as_array, x_coord, y_coord):
    top_six = []
    nine_pix = []
    value = 0

    #put the 9 pixels in the 3x3 square into
    for y_get in range(y_coord-1,y_coord+2):
        for x_get in range(x_coord-1,x_coord+2):
            nine_pix.append(imGray_as_array[y_get][x_get])
            value += imGray_as_array[y_get][x_get]

    for y in range(1, len(imGray_as_array) - 1):
        row = imGray_as_array[y]
        for x in range(int(round((len(row))/2)), len(row) - 1):
            difference_count = 0
            # count = 0
            # for y2 in range(y-1,y+2):
            #     for x2 in range(x-1,x+2):
            #         #gray_nine= 0.21*(nine_pix[count][0])+0.72*(nine_pix[count][1])+0.07**(nine_pix[count][2])
            #         #gray_array= 0.21*(imGray_as_array[y2][x2][0])+0.72*(imGray_as_array[y2][x2][1])+0.07**(imGray_as_array[y2][x2][2])
            #         difference_count =difference_count+abs(nine_pix[count] - imGray_as_array[y2][x2])
            #         count += 1

            current_patch = training_data[y][x]
            for i in range(9):
                difference_count += abs(nine_pix[i] - current_patch[i])

            if(len(top_six)<6):
                top_six.append((difference_count,(x,y)))
            else:
                for element in top_six:
                    if(element[0]>difference_count):
                        top_six.remove(element)
                        top_six.append((difference_count, (x, y)))
                        top_six.sort(key=operator.itemgetter(0))
                        break
    top_six.sort(key=operator.itemgetter(0))
    return top_six