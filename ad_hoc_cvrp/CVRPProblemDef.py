import torch
import pandas as pd
import matplotlib.pyplot as plt
import os

def get_random_problems(batch_size, min_problem_size, max_problem_size, min_agent_num, max_agent_num, random_seed=None):

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'training_data.csv')

    data = pd.read_csv(file_path)

    data_tensor = torch.tensor(data.values)

    scenario_scale = 10000.0

    max_x = 12730000
    min_x = 12650000
    max_y = 2590000
    min_y = 2550000

    if random_seed != None:
        torch.manual_seed(random_seed)


    while True:
        x_range = torch.randint(min_x, max_x - int(scenario_scale), (1,))[0]

        y_range = torch.randint(min_y, max_y - int(scenario_scale), (1,))[0]

        indices = torch.where((data_tensor[:, 0] >= x_range) & (data_tensor[:, 0] <= x_range + scenario_scale) & (data_tensor[:, 1] >= y_range) & (data_tensor[:, 1] <= y_range + scenario_scale))[0]

        if len(indices) >= 3000:
            break

    min_ = torch.min(data_tensor[indices], dim=0).values
    max_ = torch.max(data_tensor[indices], dim=0).values

    # print(min_, max_)

    problem_size = torch.randint(min_problem_size, max_problem_size + 1, size=(1, 1))[0][0]

    # print(problem_size)

    sample_index_index = torch.randint(0, len(indices), size=(batch_size, problem_size + 1))

    sample_index = indices[sample_index_index]

    sampled_data_original = data_tensor[sample_index]

    sampled_data = sampled_data_original.clone().float()

    sampled_data[:, :, :2] = (sampled_data_original[:, :, :2] - min_[:2][None, None, :]) / scenario_scale

    depot_xy = sampled_data[:, 0, :2].unsqueeze(1).to(torch.float32)

    node_xy = sampled_data[:, 1:, :2].to(torch.float32)

    agent_num = torch.randint(min_agent_num, max_agent_num + 1, size=(1, 1))[0][0]

    node_demand = (sampled_data[:, 1:, 2] * 0.01).to(torch.float32)

    # print(torch.sum(node_demand, -1))

    agent_capacity = torch.rand(size=(batch_size, agent_num)) * 3 + 2

    agent_speed = torch.rand(size=(batch_size, agent_num)) * 2 + 2

    return depot_xy, node_xy, node_demand, agent_capacity, agent_speed

def get_random_problems_normal(batch_size, min_problem_size, max_problem_size, min_agent_num, max_agent_num, random_seed=None):

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'training_data.csv')

    data = pd.read_csv(file_path)

    data_tensor = torch.tensor(data.values)

    scenario_scale = 10000.0

    max_x = 12730000
    min_x = 12650000
    max_y = 2590000
    min_y = 2550000

    if random_seed != None:
        torch.manual_seed(random_seed)


    while True:
        x_range = torch.randint(min_x, max_x - int(scenario_scale), (1,))[0]

        y_range = torch.randint(min_y, max_y - int(scenario_scale), (1,))[0]

        indices = torch.where((data_tensor[:, 0] >= x_range) & (data_tensor[:, 0] <= x_range + scenario_scale) & (data_tensor[:, 1] >= y_range) & (data_tensor[:, 1] <= y_range + scenario_scale))[0]

        if len(indices) >= 3000:
            break

    min_ = torch.min(data_tensor[indices], dim=0).values
    max_ = torch.max(data_tensor[indices], dim=0).values

    # print(min_, max_)

    problem_size = torch.randint(min_problem_size, max_problem_size + 1, size=(1, 1))[0][0]

    # print(problem_size)

    sample_index_index = torch.randint(0, len(indices), size=(batch_size, problem_size + 1))

    sample_index = indices[sample_index_index]

    sampled_data_original = data_tensor[sample_index]

    sampled_data = sampled_data_original.clone().float()

    sampled_data[:, :, :2] = (sampled_data_original[:, :, :2] - min_[:2][None, None, :]) / scenario_scale

    depot_xy = sampled_data[:, 0, :2].unsqueeze(1).to(torch.float32)

    node_xy = sampled_data[:, 1:, :2].to(torch.float32)

    agent_num = torch.randint(min_agent_num, max_agent_num + 1, size=(1, 1))[0][0]

    node_demand = (sampled_data[:, 1:, 2] * 0.01).to(torch.float32)

    # print(torch.sum(node_demand, -1))

    agent_capacity = torch.clamp(torch.randn(size=(batch_size, agent_num)) + 3.5, min=1)

    agent_speed = torch.clamp(torch.randn(size=(batch_size, agent_num)) + 3, min=1)

    return depot_xy, node_xy, node_demand, agent_capacity, agent_speed


def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data

if __name__ == '__main__':
    get_random_problems(10, 50, 80, 3, 8)