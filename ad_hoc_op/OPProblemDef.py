import torch
import pandas as pd
import matplotlib.pyplot as plt
import os

def get_random_problems(batch_size, min_problem_size, max_problem_size, min_agent_num, max_agent_num, random_seed=None):

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'training_data.csv')

    data = pd.read_csv(file_path)

    data_tensor = torch.tensor(data.values)

    if random_seed != None:
        torch.manual_seed(random_seed)

    problem_size = torch.randint(min_problem_size, max_problem_size + 1, size=(1, 1))[0][0]

    sample_index = torch.randint(0, data_tensor.size(0), size=(batch_size, problem_size + 1))

    sampled_data = data_tensor[sample_index]

    depot_xy = sampled_data[:, 0, :2].unsqueeze(1).to(torch.float32)

    node_xy = sampled_data[:, 1:, :2].to(torch.float32)

    agent_num = torch.randint(min_agent_num, max_agent_num + 1, size=(1, 1))[0][0]

    node_prize = sampled_data[:, 1:, 2].to(torch.float32)

    agent_speed = (torch.rand(size=(batch_size, agent_num)) * 50 + 50) / 101

    max_time = torch.rand(size=(batch_size, agent_num)) * 2 + 2

    # for i in range(len(node_xy[0])):
    #     plt.scatter(node_xy[0, i, 0], node_xy[0, i, 1], marker='x', s=100, c='red')
    #
    # plt.title('Clustered Data with Normalized Centers')
    # plt.xlabel('Normalized X')
    # plt.ylabel('Normalized Y')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('d')
    # plt.show()


    return depot_xy, node_xy, node_prize, agent_speed, max_time

def get_random_problems_normal(batch_size, min_problem_size, max_problem_size, min_agent_num, max_agent_num, random_seed=None):

    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'training_data.csv')

    data = pd.read_csv(file_path)

    data_tensor = torch.tensor(data.values)

    if random_seed != None:
        torch.manual_seed(random_seed)

    problem_size = torch.randint(min_problem_size, max_problem_size + 1, size=(1, 1))[0][0]

    sample_index = torch.randint(0, data_tensor.size(0), size=(batch_size, problem_size + 1))

    sampled_data = data_tensor[sample_index]

    depot_xy = sampled_data[:, 0, :2].unsqueeze(1).to(torch.float32)

    node_xy = sampled_data[:, 1:, :2].to(torch.float32)

    agent_num = torch.randint(min_agent_num, max_agent_num + 1, size=(1, 1))[0][0]

    node_prize = sampled_data[:, 1:, 2].to(torch.float32)

    agent_speed = torch.clamp(torch.randn(size=(batch_size, agent_num)) * 20 + 75, min=40) / 101

    max_time = torch.clamp(torch.randn(size=(batch_size, agent_num)) + 3, min=1)

    return depot_xy, node_xy, node_prize, agent_speed, max_time


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

if __name__ == "__main__":
    get_random_problems(2, 100, 120, 5, 10)