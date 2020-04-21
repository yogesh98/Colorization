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
    while len(clustering_attempts) <= num_attempts:
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
        centroid_changed = True
        clusters = []; for i in centers: clusters.append([])
        while centroid_changed:
            centroid_changed = False
            for row in im:
                for current in row:
                    closest_centroid_indices = []
                    smallest_distance = float('inf')
                    for i in range(len(centers)):
                        centroid = centers[i]
                        distance = euclidean_distance(current, centroid)

                        if distance <= smallest_distance:
                            smallest_distance = distance
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
            variances.append(variance(cluster))
        clustering_attempts.append((centers, sum(variances)))

    min_variance = float('inf')
    min_clusters = []
    for attempt in clustering_attempts:
        cluster = attempt[0]
        var = attempt[1]
        if var < min_variance:
            min_clusters.append(cluster)

    return random.choice(min_clusters)

def euclidean_distance(start, end):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(start, end)]))


def mean_point(lst):
    return (int(round(sum([point[0] for point in lst])/len(lst))),
            int(round(sum([point[1] for point in lst])/len(lst))),
            int(round(sum([point[2] for point in lst])/len(lst))))


def variance(lst):
    mean = sum(lst)/len(lst)

    return sum((i - mean) ** 2 for i in lst) / len(lst)