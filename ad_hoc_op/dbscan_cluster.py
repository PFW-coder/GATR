import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np

def cluster(filename):
    data = pd.read_csv(filename)

    x = data['bd09mc_X'].values
    y = data['bd09mc_Y'].values
    values_ = data['value'].values

    coordinates_ = np.column_stack((x, y))

    coordinates = []
    values = []

    for i in range(len(coordinates_)):
        if coordinates_[i, 1] > 2530000:
            coordinates.append(coordinates_[i, :])
            values.append(values_[i])
    coordinates = np.array(coordinates)
    values = np.array(values)

    max_x = np.max(coordinates[:, 0]) - np.min(coordinates[:, 0])
    max_y = np.max(coordinates[:, 1]) - np.min(coordinates[:, 1])

    max_scale = max(max_x, max_y)
    print(max_scale)
    normalized_coordinates = coordinates / max_scale
    normalized_coordinates[:, 0] = normalized_coordinates[:, 0] - np.min(normalized_coordinates[:, 0])
    normalized_coordinates[:, 1] = normalized_coordinates[:, 1] - np.min(normalized_coordinates[:, 1])

    a = 500.0 / max_scale
    min_samples = 20

    db = DBSCAN(eps=a, min_samples=min_samples).fit(normalized_coordinates)
    labels = db.labels_

    cluster_centers = []
    total_values = []

    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            continue
        class_member_mask = (labels == label)
        cluster_points = normalized_coordinates[class_member_mask]
        cluster_values = values[class_member_mask]

        cluster_center = cluster_points.mean(axis=0)
        total_value = cluster_values.sum()
        cluster_centers.append(cluster_center)
        total_values.append(total_value)

    cluster_centers_new = []
    total_values_new = []

    largest_value = max(total_values)
    for i in range(len(cluster_centers)):
        normalized_value = total_values[i] / largest_value
        cluster_centers_new.append(cluster_centers[i])
        total_values_new.append(normalized_value)

    print(len(cluster_centers_new))
    return cluster_centers_new, total_values_new

