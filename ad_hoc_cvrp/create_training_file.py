import pandas as pd
import numpy as np
import csv

combined_data = pd.DataFrame(columns=['X', 'Y', 'Value'])

x_list = []
y_list = []
value_list = []
for i in range(100, 110):
    filename = '/ad-hoc-team/data/SZ_2023100' + str(i) + '.csv'
    data = pd.read_csv(filename)
    x = data['bd09mc_X'].values
    y = data['bd09mc_Y'].values
    values = data['value'].values
    x_list += x.tolist()
    y_list += y.tolist()
    value_list += values.tolist()

x_list_new = []
y_list_new = []
value_list_new = []

for i in range(len(x_list)):
    if y_list[i] > 2530000 and value_list[i] <= 50:
        x_list_new.append(x_list[i])
        y_list_new.append(y_list[i])
        value_list_new.append(value_list[i])

x_ = np.array(x_list_new)
y_ = np.array(y_list_new)
value_ = np.array(value_list_new)
value_ = value_

csv_file = "training_data.csv"

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)

    writer.writerow(['x', 'y', 'value'])

    for x, y, value in zip(x_, y_, value_):
        writer.writerow([x, y, value])

print("Write Successfully")