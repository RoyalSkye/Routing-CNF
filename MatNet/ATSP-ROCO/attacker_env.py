import torch
import random
import os
import glob
from copy import deepcopy
import time
import numpy as np
import tsplib95
from ATSP_algorithms import calc_lkh_tour_len, calc_nearest_neighbor_tour_len, solveFarthestInsertion, get_adj, calc_MatNet_tour_len, calc_furthest_insertion_tour_len

VERY_LARGE_INT = 10  # 65536


def parse_tsp(list_m, dim=None, name='unknown'):
    if dim is None:
        dim = len(list_m)
    outstr = ''
    # outstr += 'NAME: %s\n' % name #problem.name
    outstr += 'TYPE: ATSP\n'
    # outstr += 'COMMENT: %s\n' % name
    outstr += 'DIMENSION: %d\n' % dim #problem.dimension
    outstr += 'EDGE_WEIGHT_TYPE: EXPLICIT\n'
    # outstr += 'EDGE_WEIGHT_FORMAT: LOWER_DIAG_ROW\n'
    outstr += 'EDGE_WEIGHT_FORMAT: FULL_MATRIX\n'
    outstr += 'EDGE_WEIGHT_SECTION:\n'
    for l in list_m:
        listToStr = ' '.join([str(elem) for elem in l])
        outstr += ' %s\n' % listToStr
    #outstr += 'EDGE_DATA_FORMAT: EDGE_LIST\n'
    #outstr += 'EDGE_DATA_SECTION:\n'
    #for edge_idx, weight in edges_dict.items():
    #    outstr += f' {edge_idx[0]+1} {edge_idx[1]+1} {weight}\n'
    #outstr += '-1\n'

    return outstr


class AttackerEnv(object):
    def __init__(self, solver_type='nn', node_dimension=20, is_attack=False, tester=None, path="./tmp", printinfo=True):
        self.solver_type = solver_type
        self.node_dimension = node_dimension
        self.process_dataset(path, printinfo)
        self.available_solvers = ('nn', 'furthest', 'lkh-5', 'MatNet')
        self.is_attack = is_attack
        self.tester = tester
        self.device = torch.device("cuda:{}".format(torch.cuda.current_device()) if torch.cuda.is_available() else "cpu")
        assert solver_type in self.available_solvers

    def process_dataset(self, dirpath="./tmp", printinfo=True):
        """
        Downloading and processing dataset.
        """
        # print("\nPreparing dataset.\n")
        self.tspfiles = []

        for fp in sorted(glob.iglob(os.path.join(dirpath, "*.atsp"))):
            self.tspfiles.append(fp)
            if printinfo:
                print(fp)

        if printinfo:
            print(f'Total file num {len(self.tspfiles)}')

    def generate_tuples(self, num_train_samples, num_test_samples, rand_id, defense=False):
        random.seed(int(rand_id))
        np.random.seed(int(rand_id + num_train_samples))
        assert num_train_samples + num_test_samples <= len(self.tspfiles), \
            f'{num_train_samples + num_test_samples} > {len(self.tspfiles)}'

        training_tuples = []
        testing_tuples = []
        sum_num_nodes = 0

        if defense:
            for i in range(num_test_samples):
                tsp_path = self.tspfiles[i]
                print(tsp_path)
                problem = tsplib95.load(tsp_path)
                # lower_left_matrix = get_lower_matrix_tsp(problem)
                lower_left_matrix = get_adj(problem)
                tsp_solutions = {}
                tsp_times = {}
                for key in self.available_solvers:
                    tour, sol, sec = self.solve_feasible_tsp(lower_left_matrix, key)
                    tsp_solutions[key] = sol
                    tsp_times[key] = sec
                    if key == self.solver_type:
                        edge_candidates = self.edge_candidate_from_tour(tour, problem.dimension)
                print(f'id {i} {problem.dimension} '
                      f'{"; ".join([f"{x} tour={tsp_solutions[x]:.4f} time={tsp_times[x]:.2f}" for x in self.available_solvers])}')
                sum_num_nodes += problem.dimension
                testing_tuples.append((
                    lower_left_matrix,
                    edge_candidates,
                    tsp_solutions[self.solver_type],
                    tsp_solutions,
                    tsp_times,
                    tsp_path
                ))
            for solver_name in self.available_solvers:
                print(f'{solver_name} average tour_len='
                      f'{torch.mean(torch.tensor([tup[3][solver_name] for tup in testing_tuples], dtype=torch.float)):.4f}')
        else:
            return_tuples = training_tuples
            for i, tsp_path in enumerate(self.tspfiles):
                problem = tsplib95.load(tsp_path)
                # lower_left_matrix = get_lower_matrix_tsp(problem)
                lower_left_matrix = get_adj(problem)
                tsp_solutions = {}
                tsp_times = {}
                for key in self.available_solvers:
                    tour, sol, sec = self.solve_feasible_tsp(lower_left_matrix, key)
                    tsp_solutions[key] = sol
                    tsp_times[key] = sec
                    if key == self.solver_type:
                        if self.is_attack:
                            edge_candidates = self.edge_candidate_Attack(tour, lower_left_matrix)
                        else:
                            edge_candidates = self.edge_candidate_from_tour(tour, problem.dimension)
                print(f'id {i} {problem.dimension} '
                      f'{"; ".join([f"{x} tour={tsp_solutions[x]:.2f} time={tsp_times[x]:.2f}" for x in self.available_solvers])}')
                sum_num_nodes += problem.dimension
                # print(lower_left_matrix, '\n\n')

                return_tuples.append((
                    lower_left_matrix,  # lower-left triangle of adjacency matrix
                    edge_candidates,  # edge candidates
                    tsp_solutions[self.solver_type],  # reference TSP solution
                    tsp_solutions,  # all TSP solutions
                    tsp_times,  # TSP solving time
                    tsp_path  # load path
                ))
                if i == num_train_samples - 1 or i == num_train_samples + num_test_samples - 1:
                    print(f'average number of nodes: {sum_num_nodes / len(return_tuples)}')
                    sum_num_nodes = 0
                    for solver_name in self.available_solvers:
                        print(f'{solver_name} average tour_len='
                              f'{torch.mean(torch.tensor([tup[3][solver_name] for tup in return_tuples], dtype=torch.float)):.4f}')
                    return_tuples = testing_tuples

        return training_tuples, testing_tuples

    def step(self, list_lower_matrix, act, prev_solution, defense=False):
        if self.is_attack and not defense:
            return self.step_Attack(list_lower_matrix, act, prev_solution)

        new_list_lower_matrix = deepcopy(list_lower_matrix)
        if isinstance(act, torch.Tensor):
            act = (act[0].item(), act[1].item())
        # if act[0] >= act[1]:
        #     idx0, idx1 = act[0], act[1]
        # else:
        #     idx0, idx1 = act[1], act[0]
        # new_list_lower_matrix[idx0][idx1] += VERY_LARGE_INT
        new_list_lower_matrix[act[0]][act[1]] *= 2
        new_tour, new_solution, _ = self.solve_feasible_tsp(new_list_lower_matrix, self.solver_type)
        new_edge_candidate = self.edge_candidate_from_tour(new_tour, len(new_list_lower_matrix))
        reward = prev_solution - new_solution
        done = new_solution == 0
        # done = False
        return reward, new_list_lower_matrix, new_edge_candidate, new_solution, done

    def step_Attack(self, list_lower_matrix, act, prev_solution):
        new_list_lower_matrix = deepcopy(list_lower_matrix)
        if isinstance(act, torch.Tensor):
            act = (act[0].item(), act[1].item())
        # if act[0] >= act[1]:
        #     idx0, idx1 = act[0], act[1]
        # else:
        #     idx0, idx1 = act[1], act[0]
        # new_list_lower_matrix[idx0][idx1] = 1
        new_list_lower_matrix[act[0]][act[1]] /= 2

        new_tour, new_solution, _ = self.solve_feasible_tsp(new_list_lower_matrix, self.solver_type)
        new_edge_candidate = self.edge_candidate_Attack(new_tour, new_list_lower_matrix)
        reward = new_solution - prev_solution
        done = new_solution == 0
        # done = False
        return reward, new_list_lower_matrix, new_edge_candidate, new_solution, done

    def step_e2e(self, list_lower_matrix, prob_dim, prob_name, act, prev_solution):
        raise NotImplementedError

    def solve_feasible_tsp(self, lower_left_matrix, solver_type):
        prev_time = time.time()
        tsp_inst = tsplib95.parse(parse_tsp(lower_left_matrix))
        if solver_type == 'nn':
            tour, length = calc_nearest_neighbor_tour_len(tsp_inst)
        elif solver_type == 'furthest':
            tour, length = solveFarthestInsertion(tsp_inst)
        elif solver_type == 'lkh-5':
            tour, length = calc_lkh_tour_len(tsp_inst)
        elif 'lkh-' in solver_type:
            num_moves = int(solver_type.strip('lkh-'))
            tour, length = calc_lkh_tour_len(tsp_inst, move_type=num_moves, runs=1)
        elif solver_type == 'MatNet':
            tour, length = calc_MatNet_tour_len((torch.tensor(lower_left_matrix) / 1e4).to(torch.float32).to(self.device), self.tester)
        else:
            raise ValueError(f'{solver_type} is not implemented.')
        comp_time = time.time() - prev_time
        return tour, length, comp_time

    @staticmethod
    def edge_candidate_from_tour(tour, num_nodes):
        assert tour[0] == tour[-1]

        edge_candidate = {x: set() for x in range(num_nodes)}
        iter_obj = iter(tour)
        last_node = next(iter_obj)
        for node in iter_obj:
            # print(last_node, node)
            edge_candidate[last_node].add(node)
            edge_candidate[node].add(last_node)
            last_node = node

        return edge_candidate

    def edge_candidate_Attack(self, tour, list_lower_matrix):
        num_nodes = len(list_lower_matrix)
        edge_candidate = {x: set() for x in range(num_nodes)}

        for i in range(num_nodes):
            for j in range(i):
                if list_lower_matrix[i][j] > 1:
                    edge_candidate[i].add(j)
                    edge_candidate[j].add(i)

        iter_obj = iter(tour)
        last_node = next(iter_obj)
        for node in iter_obj:
            if node in edge_candidate[last_node]:
                edge_candidate[last_node].remove(node)
            if last_node in edge_candidate[node]:
                edge_candidate[node].remove(last_node)
            last_node = node

        return edge_candidate
