import math
import numpy as np
import matplotlib.pyplot as plt
import csv
import torch

def latlon_to_local(lat, lon, lat0, lon0):
    R = 6371000
    x = R * (math.radians(lon) - math.radians(lon0)) * math.cos(math.radians(lat0))
    y = R * (math.radians(lat) - math.radians(lat0))
    return x, y

lat0, lon0 = 22.5314, 113.9294

content = np.loadtxt('long_lat_test.txt', delimiter=',')

loc_ = np.zeros_like(content)

for i in range(len(loc_)):
    x, y = latlon_to_local(content[i][1], content[i][0], lat0, lon0)
    loc_[i][0], loc_[i][1] = x, y

scale_ = np.max(np.max(loc_, axis=0) - np.min(loc_, axis=0))
min_ = np.min(loc_, axis=0)
print(scale_)

loc_[:, 0] = (loc_[:, 0] - min_[0]) / 11000
loc_[:, 1] = (loc_[:, 1] - min_[1]) / 11000

xy_ = torch.tensor(loc_)

agent_speed = [50.0 / 11, 50.0 / 11, 50.0 / 11, 50.0 / 11, 80.0 / 11, 80.0 / 11]
max_time = [2.5, 2.5, 2.5, 2.5, 3.0, 3.0]

filename = 'data/test_data.pt'

torch.save({
    'depot_xy': xy_[0].unsqueeze(0).unsqueeze(0).to(torch.float32),
    'node_xy': xy_[1:].unsqueeze(0).to(torch.float32),
    'agent_speed': torch.tensor(agent_speed).unsqueeze(0).to(torch.float32),
    'max_time': torch.tensor(max_time).unsqueeze(0).to(torch.float32)
}, filename)