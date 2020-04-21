from PIL import Image
import random
import math
# Takes Image object and returns a 2d list of RGB values as tuples in the lists
def image_to_2d_array(im):
    w = im.size[0]
    h = im.size[1]

    arr = list(im.getdata())

    return [arr[i:i + w] for i in range(0, len(arr), w)]

# takes 2d list of RGB tuples and performs a k means clustering on them
def k_means_clustering_on_img(k, im, num_attempts):
    clustering_attempts = []
    while len(clustering_attempts) <= num_attempts
        # Randomly selecting starting clusters
        centers = []
        while len(centers) <= 5:
            choice = random.choice(random.choice(im))
            if choice not in centers:
                centers.append(choice)

        # group points in clusters by selecting nearest cluster
        # calculate the mean of each cluster, and recluster using mean values
        # keep doing this until mean values do not change
        # also keep track of distances in another list so its easier to calculate variation later
        centriod_changed = True
        clusters = []; for i in centers: clusters.append([])
        while centriod_changed:
            for row in im:
                for current in row:
                    closest_centroid_indices = []
                    smallest_distance = float('inf')
                    for i in range(len(centers)):
                        centroid = centers[i]
                        distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(current, centroid)]))

                        if distance <= smallest_distance:
                            smallest_distance = distance
                            closest_centroid_indices.append(i)
