import pandas as pd
from sklearn.cluster import DBSCAN
import hdbscan
import numpy as np
import torch
import matplotlib.pyplot as plt

def dbscan_test(filename, created_file):
    data = pd.read_csv(filename)
    x = data['bd09mc_X'].values
    y = data['bd09mc_Y'].values
    values_ = data['value'].values

    # 将x和y坐标组合成一个二维数组
    coordinates_ = np.column_stack((x, y))

    coordinates = []
    values = []

    for i in range(len(coordinates_)):
        if coordinates_[i, 1] > 2530000:
            coordinates.append(coordinates_[i, :])
            values.append(values_[i])
    coordinates = np.array(coordinates)
    values = np.array(values)

    max_scale = 101000
    normalized_coordinates = coordinates / max_scale
    normalized_coordinates[:, 0] = normalized_coordinates[:, 0] - np.min(normalized_coordinates[:, 0])
    normalized_coordinates[:, 1] = normalized_coordinates[:, 1] - np.min(normalized_coordinates[:, 1])


    a = 500.0 / max_scale
    min_samples = 20

    db = DBSCAN(eps=a, min_samples=min_samples).fit(normalized_coordinates)
    labels = db.labels_

    # clusterer = hdbscan.HDBSCAN(min_cluster_size=10)  # 你可以调整min_cluster_size参数
    # labels = clusterer.fit_predict(normalized_coordinates)

    cluster_centers = []
    total_values = []

    unique_labels = set(labels)
    for label in unique_labels:
        if label == -1:
            # -1表示噪声点，不属于任何簇
            continue
        # 获取当前簇的所有点
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
        if normalized_value > 0.0:
            cluster_centers_new.append(cluster_centers[i])
            total_values_new.append(normalized_value)

    agent_speed = [50.0 / 101, 80.0 / 101, 80.0 / 101, 80.0 / 101, 100.0 / 101]
    max_time = [2.0, 2.5, 2.5, 2.5, 5.0]

    torch.save({
        'depot_xy': torch.tensor(cluster_centers_new)[0].unsqueeze(0).unsqueeze(0).to(torch.float32),
        'node_xy': torch.tensor(cluster_centers_new).unsqueeze(0).to(torch.float32),
        'node_prize': torch.tensor(total_values_new).unsqueeze(0).to(torch.float32),
        'agent_speed': torch.tensor(agent_speed).unsqueeze(0).to(torch.float32),
        'max_time': torch.tensor(max_time).unsqueeze(0).to(torch.float32)
    }, created_file)

    # print(len(cluster_centers_new))
    return(len(cluster_centers_new))


if __name__ == "__main__":

    list = []

    for i in range(110, 124):
        filename = '/ad-hoc-team/data/SZ_2023100' + str(i) + '.csv'
        created_file = "data/test_data_" + str(i) + ".pt"
        a = dbscan_test(filename, created_file)
        list.append(a)

    print(list)



