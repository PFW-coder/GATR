
from dataclasses import dataclass
import torch

from CVRPProblemDef import get_random_problems, augment_xy_data_by_8_fold, get_random_problems_normal


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)
    agent_capacity: torch.Tensor = None
    # shape: (batch, agent)
    agent_speed: torch.Tensor = None
    # shape: (batch, agent)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, pomo, agent)
    current_node: torch.Tensor = None
    # shape: (batch, pomo, agent)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, agent, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo, agent)


class CVRPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.max_problem_size = env_params['max_problem_size']
        self.min_problem_size = env_params['min_problem_size']
        self.pomo_size = env_params['pomo_size']
        self.max_agent_num = env_params['max_agent_num']
        self.min_agent_num = env_params['min_agent_num']

        self.FLAG__use_saved_problems = False
        self.FLAG__use_random_seed = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_agent_capacity = None
        self.saved_agent_speed = None
        self.saved_index = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo, agent)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)
        self.agent_capacity = None
        # shape: (batch, agent)
        self.agent_speed = None
        # shape: (batch, agent)

        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo, agent)
        self.selected_node_list = None
        # shape: (batch, pomo, agent, 0~)

        self.at_the_depot = None
        # shape: (batch, pomo, agent)
        self.load = None
        # shape: (batch, pomo, agent)
        self.used_time = None
        # shape: (batch, pomo, agent)
        self.arrival_time = None
        # shape: (batch, pomo, problem+1)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, agent, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, agent, problem+1)
        self.finished = None
        # shape: (batch, pomo, agent)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def use_saved_problems(self, filename, device):
        self.FLAG__use_saved_problems = True
        loaded_dict = torch.load(filename, map_location=device)
        self.saved_depot_xy = loaded_dict['depot_xy']
        self.saved_node_xy = loaded_dict['node_xy']
        self.saved_node_demand = loaded_dict['node_demand']
        self.saved_agent_capacity = loaded_dict['agent_capacity']
        self.saved_agent_speed = loaded_dict['agent_speed']
        self.saved_index = 0

    def set_random_seed(self, random_seed, test_num):
        self.FLAG__use_random_seed = True
        torch.manual_seed(random_seed)
        self.random_list = torch.randint(0, 100000, size=(test_num, 1)).squeeze(-1)
        self.random_list_index = 0
        torch.seed()

    def load_problems(self, batch_size, aug_factor=1):
        self.batch_size = batch_size

        if (not self.FLAG__use_saved_problems) and (not self.FLAG__use_random_seed):
            depot_xy, node_xy, node_demand, agent_capacity, agent_speed = get_random_problems(batch_size, self.min_problem_size, self.max_problem_size, self.min_agent_num, self.max_agent_num)
        elif self.FLAG__use_random_seed:
            depot_xy, node_xy, node_demand, agent_capacity, agent_speed = get_random_problems(batch_size,
                                                                                              self.min_problem_size,
                                                                                              self.max_problem_size,
                                                                                              self.min_agent_num,
                                                                                              self.max_agent_num,
                                                                                              self.random_list[self.random_list_index])
        else:
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index+batch_size]
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index+batch_size]
            node_demand = self.saved_node_demand[self.saved_index:self.saved_index+batch_size]
            agent_capacity = self.saved_agent_capacity[self.saved_index:self.saved_index+batch_size]
            agent_speed = self.saved_agent_speed[self.saved_index:self.saved_index+batch_size]
            self.saved_index += batch_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
                agent_capacity = agent_capacity.repeat(8, 1)
                agent_speed = agent_speed.repeat(8, 1)
            else:
                raise NotImplementedError

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(node_demand.size(0), 1))
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)

        self.batch_size = node_demand.size(0)
        self.agent_num = agent_capacity.size(1)
        self.problem_size = node_xy.size(1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None, None].expand(self.batch_size, self.pomo_size, self.agent_num)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :, None].expand(self.batch_size, self.pomo_size, self.agent_num)
        self.AGENT_IDX = torch.arange(self.agent_num)[None, None, :].expand(self.batch_size, self.pomo_size,
                                                                           self.agent_num)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand
        self.reset_state.agent_capacity = agent_capacity
        self.reset_state.agent_speed = agent_speed

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        self.step_state.AGENT_IDX = self.AGENT_IDX
        self.step_state.graph_size = self.depot_node_xy.size(-2)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo, agent)

        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, self.agent_num, 0), dtype=torch.long)
        # shape: (batch, pomo, agent, 0~)
        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size, self.agent_num), dtype=torch.bool)
        # shape: (batch, pomo, agent)
        self.agent_capacity = self.reset_state.agent_capacity
        self.load = self.reset_state.agent_capacity[:, None, :].clone().repeat(1, self.pomo_size, 1)
        # shape: (batch, pomo, agent)
        self.used_time = torch.zeros(size=(self.batch_size, self.pomo_size, self.agent_num))
        # shape: (batch, pomo, agent)
        self.arrival_time = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.agent_speed = self.reset_state.agent_speed
        # shape: (batch, agent)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.agent_num, self.problem_size+1))
        # shape: (batch, pomo, agent, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.agent_num, self.problem_size+1))
        # shape: (batch, pomo, agent, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size, self.agent_num), dtype=torch.bool)
        # shape: (batch, pomo, agent)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.used_time = self.used_time
        self.step_state.agent_capacity = self.agent_capacity
        self.step_state.agent_speed = self.agent_speed
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo, agent)
        # Dynamic-1
        ####################################
        self.selected_count += 1
        if self.current_node is None:
            last_node = selected.clone()
        else:
            last_node = self.current_node.clone()
        # shape: (batch, pomo, agent)
        self.current_node = selected
        # shape: (batch, pomo, agent)

        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, :, None]), dim=3)
        # shape: (batch, pomo, agent, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)

        demand_list = self.depot_node_demand[:, None, None, :].expand(self.batch_size, self.pomo_size, self.agent_num, -1)
        # shape: (batch, pomo, agent, problem+1)

        change_node = torch.where(last_node != selected, selected, 0)
        # shape: (batch, pomo, agent)
        gathering_index = change_node[:, :, :, None]
        # shape: (batch, pomo, agent, 1)
        selected_demand = demand_list.gather(dim=3, index=gathering_index).squeeze(dim=3)

        self.load -= selected_demand
        # shape: (batch, pomo, agent)

        self.load[self.at_the_depot] = self.reset_state.agent_capacity[:, None, :].clone().repeat(1, self.pomo_size, 1)[self.at_the_depot]
        # refill loaded at the depot

        all_xy = self.depot_node_xy[:, None, None, :, :].repeat(1, self.pomo_size, self.agent_num, 1, 1)
        # shape: (batch, pomo, agent, problem+1, 2)

        last_node_xy = all_xy.gather(dim=3, index=last_node[:, :, :, None, None].repeat(1, 1, 1, 1, 2))
        # shape: (batch, pomo, agent, 1, 2)

        current_node_xy = all_xy.gather(dim=3, index=self.current_node[:, :, :, None, None].repeat(1, 1, 1, 1, 2))
        # shape: (batch, pomo, agent, 1, 2)

        # self.used_time += ((current_node_xy-last_node_xy)**2).sum(4).sqrt().sum(3) / (self.agent_speed[:, None, :].repeat(1, self.pomo_size, 1))
        self.used_time += torch.abs(current_node_xy-last_node_xy).sum(4).sum(3) / (
            self.agent_speed[:, None, :].repeat(1, self.pomo_size, 1))
        # shape: (batch, pomo, agent)

        self.arrival_time = self.arrival_time.scatter(-1, self.current_node, self.used_time)

        service_time_index = torch.where(last_node == self.current_node, 0, 1)
        service_time_index = torch.where(self.current_node == 0, 0, service_time_index)
        self.used_time += service_time_index * (1.0 / 6.0)

        load_time_index = torch.where(last_node == self.current_node, 0, 1)
        load_time_index = torch.where(self.current_node == 0, load_time_index, 0)
        self.used_time += load_time_index * (1.0 / 6.0)

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, self.AGENT_IDX, selected] = float('-inf')
        # shape: (batch, pomo, agent, problem+1)

        temp_visited_ninf_flag = self.visited_ninf_flag[:, :, :, 1:].clone()
        # shape: (batch, pomo, agent, problem)

        self.visited_ninf_flag[:, :, :, 1:] = temp_visited_ninf_flag.sum(-2)[:, :, None, :].repeat(1, 1, self.agent_num, 1)
        # shape: (batch, pomo, agent, problem+1)

        self.visited_ninf_flag[:, :, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, :, None] + round_error_epsilon < demand_list
        # shape: (batch, pomo, agent, problem+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, pomo, agent, problem+1)

        if (self.load < -2 * round_error_epsilon).any():
            raise ValueError("wrong")

        newly_finished = (self.visited_ninf_flag[:, :, :, 1:] == float('-inf')).all(dim=3)
        # shape: (batch, pomo, agent)

        self.finished = self.finished + newly_finished
        # shape: (batch, pomo, agent)

        self.ninf_mask[:, :, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.used_time = self.used_time
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_max_arrival_time()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_max_arrival_time(self):
        max_arrival_time = torch.max(self.arrival_time[:, :, 1:], -1).values
        # shape: (batch, pomo)
        return max_arrival_time


