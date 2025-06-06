import math
import numpy as np
import matplotlib.pyplot as plt
import csv

def latlon_to_local(lat, lon, lat0, lon0):
    R = 6371000
    x = R * (math.radians(lon) - math.radians(lon0)) * math.cos(math.radians(lat0))
    y = R * (math.radians(lat) - math.radians(lat0))
    return x, y

lat0, lon0 = 22.5314, 113.9294

content = np.loadtxt('long_lat.txt', delimiter=',')

loc_ = np.zeros_like(content)

for i in range(len(loc_)):
    x, y = latlon_to_local(content[i][1], content[i][0], lat0, lon0)
    loc_[i][0], loc_[i][1] = x, y

scale_ = np.max(np.max(loc_, axis=0) - np.min(loc_, axis=0))
min_ = np.min(loc_, axis=0)
print(scale_)

loc_[:, 0] = (loc_[:, 0] - min_[0]) / 11000
loc_[:, 1] = (loc_[:, 1] - min_[1]) / 11000

csv_file = "training_data.csv"

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(['x', 'y'])

    for i in range(len(loc_)):
        writer.writerow([loc_[i][0], loc_[i][1]])

print("Write Successfully")

for i in range(len(loc_)):
    plt.scatter(loc_[i][0], loc_[i][1], marker='x', s=100, c='red')

plt.title('a')
plt.xlabel('Normalized X')
plt.ylabel('Normalized Y')
plt.legend()
plt.grid(True)
plt.savefig('s')
plt.show()