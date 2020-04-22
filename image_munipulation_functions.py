from PIL import Image, ImageOps
import random
import math
from PIL import Image
# Takes Image object and returns a 2d list of RGB values as tuples in the lists
def image_to_2d_array(im):
    w = im.size[0]
    h = im.size[1]

    arr = list(im.getdata())

    return [arr[i:i + w] for i in range(0, len(arr), w)]

# takes 2d list of RGB tuples and performs a k means clustering on them
# Returns the k colors that it found in the clustering
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

def image_first_transform(colors, im, im_as_array):
    imGray = im.convert("L")
    im_as_arrayNew = image_to_2d_array(imGray)
    for y in range(len(im_as_array) - 1):
        row = im_as_array[y]
        for x in range((len(row) - 1)):
            if( x >= (len(row) - 1)/2):
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