import os, time
import pickle
import numpy as np
import torch
from logging import getLogger

from CVRPEnv import CVRPEnv as Env
from CVRPModel import CVRPModel as Model

from utils.utils import *
from utils.functions import load_dataset, save_dataset


class CVRPTester:
    def __init__(self, env_params, model_params, tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='tester')
        self.result_folder = get_result_folder()

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device
        self.model_params['device'] = self.device

        # ENV and MODEL
        self.num_expert = self.tester_params['num_expert']
        self.path_list = None
        # self.env = Env(**self.env_params)
        if self.num_expert == 1:
            self.model = Model(**self.model_params)
        else:
            self.models = [Model(**self.model_params) for _ in range(self.num_expert)]

        # load dataset
        if tester_params['test_set_path'].endswith(".pkl"):
            data = load_dataset(tester_params['test_set_path'])[: self.tester_params['test_episodes']]
            depot_xy, node_xy, node_demand, capacity = [i[0] for i in data], [i[1] for i in data], [i[2] for i in data], [i[3] for i in data]
            depot_xy, node_xy, node_demand, capacity = torch.Tensor(depot_xy).to(self.device), torch.Tensor(node_xy).to(self.device), torch.Tensor(node_demand).to(self.device), torch.Tensor(capacity).to(self.device)
            node_demand = node_demand / capacity.view(-1, 1)
            self.test_data = (depot_xy, node_xy, node_demand)
            opt_sol = load_dataset(tester_params['test_set_opt_sol_path'], disable_print=True)[: self.tester_params['test_episodes']]  # [(obj, route), ...]
            self.opt_sol = [i[0] for i in opt_sol]
        else:
            # for solving instances with TSPLIB format
            self.path_list = [os.path.join(tester_params['test_set_path'], f) for f in sorted(os.listdir(tester_params['test_set_path']))] \
                if os.path.isdir(tester_params['test_set_path']) else [tester_params['test_set_path']]
            self.opt_sol = None
            assert self.path_list[-1].endswith(".vrp")

        # Load checkpoint
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        if self.num_expert == 1:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.models = [self.model]
        else:
            model_state_dict = checkpoint['model_state_dict']
            # self.models = [self.models[i].load_state_dict(model_state_dict[i]) for i in range(self.num_expert)]  # '_IncompatibleKeys' object has no attribute 'eval'
            for i in range(self.num_expert):
                self.models[i].load_state_dict(model_state_dict[i])

        # utility
        self.time_estimator = TimeEstimator()

    def run(self, i):
        start_time = time.time()
        scores, aug_scores = torch.zeros(0), torch.zeros(0)

        if self.path_list:
            for path in self.path_list:
                score, aug_score = self._solve_cvrplib(self.models[i], path)
                scores = torch.cat((scores, score), dim=0)
                aug_scores = torch.cat((aug_scores, aug_score), dim=0)
        else:
            scores, aug_scores = self._test(self.models[i])

        print(">> Evaluation on {} finished within {:.2f}s".format(self.tester_params['test_set_path'], time.time() - start_time))

        # save results to file
        # with open(os.path.split(self.tester_params['test_set_path'])[-1], 'wb') as f:
        #     result = {"score_list": self.score_list, "aug_score_list": self.aug_score_list, "gap_list": self.gap_list, "aug_gap_list": self.aug_gap_list}
        #     pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
        #     print(">> Save final results to {}".format(os.path.split(self.tester_params['test_set_path'])[-1]))

        return scores.cpu(), aug_scores.cpu(), self.opt_sol, i

    def _test(self, model):
        self.time_estimator.reset()
        env_params = {'problem_size': self.test_data[1].size(1), 'pomo_size': self.test_data[1].size(1), 'device': self.device}
        env = Env(**env_params)
        score_AM, gap_AM = AverageMeter(), AverageMeter()
        aug_score_AM, aug_gap_AM = AverageMeter(), AverageMeter()
        scores, aug_scores = torch.zeros(0).to(self.device), torch.zeros(0).to(self.device)

        test_num_episode = self.tester_params['test_episodes']
        assert self.test_data[0].size(0) == test_num_episode, "the number of test instances does not match!"
        episode = 0
        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)
            depot_xy, node_xy, node_demand = self.test_data
            test_data = (depot_xy[episode: episode + batch_size], node_xy[episode: episode + batch_size], node_demand[episode: episode + batch_size])
            score, aug_score, all_score, all_aug_score = self._test_one_batch(model, env, test_data)
            opt_sol = self.opt_sol[episode: episode + batch_size]

            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)
            episode += batch_size
            gap = [(all_score[i].item() - opt_sol[i]) / opt_sol[i] * 100 for i in range(batch_size)]
            aug_gap = [(all_aug_score[i].item() - opt_sol[i]) / opt_sol[i] * 100 for i in range(batch_size)]
            gap_AM.update(sum(gap) / batch_size, batch_size)
            aug_gap_AM.update(sum(aug_gap) / batch_size, batch_size)
            scores = torch.cat((scores, all_score), dim=0)
            aug_scores = torch.cat((aug_scores, all_aug_score), dim=0)

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f}, Gap: {:.4f} ".format(score_AM.avg, gap_AM.avg))
                self.logger.info(" AUGMENTATION SCORE: {:.4f}, Gap: {:.4f} ".format(aug_score_AM.avg, aug_gap_AM.avg))
                print("{:.3f} ({:.3f}%)".format(score_AM.avg, gap_AM.avg))
                print("{:.3f} ({:.3f}%)".format(aug_score_AM.avg, aug_gap_AM.avg))

        return scores, aug_scores

    def _test_one_batch(self, model, env, test_data):
        """
            Note: test_data should include depot_xy, node_xy, normalized node_demand.
        """
        batch_size = test_data[0].size(0)
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        model.eval()
        with torch.no_grad():
            env.load_problems(batch_size, problems=test_data, aug_factor=aug_factor)
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)

        # POMO Rollout
        state, reward, done = env.pre_step()
        while not done:
            selected, _ = model(state)
            # shape: (batch, pomo)
            state, reward, done = env.step(selected)

        # Return
        aug_reward = reward.reshape(aug_factor, batch_size, env.pomo_size)
        # shape: (augmentation, batch, pomo)
        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = -max_pomo_reward[0, :].float()  # negative sign to make positive value
        no_aug_score_mean = no_aug_score.mean()

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float()  # negative sign to make positive value
        aug_score_mean = aug_score.mean()

        return no_aug_score_mean.item(), aug_score_mean.item(), no_aug_score, aug_score

    def _solve_cvrplib(self, model, path):
        """
            Solving one instance with CVRPLIB format.
        """
        file = open(path, "r")
        lines = [ll.strip() for ll in file]
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("DIMENSION"):
                dimension = int(line.split(':')[1])
            elif line.startswith("CAPACITY"):
                capacity = int(line.split(':')[1])
            elif line.startswith('NODE_COORD_SECTION'):
                locations = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
                i = i + dimension
            elif line.startswith('DEMAND_SECTION'):
                demand = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
                i = i + dimension
            i += 1
        original_locations = locations[:, 1:]
        original_locations = np.expand_dims(original_locations, axis=0)  # [1, n+1, 2]
        loc_scaler = 1000
        locations = original_locations / loc_scaler  # [1, n+1, 2]: Scale location coordinates to [0, 1]
        depot_xy, node_xy = torch.Tensor(locations[:, :1, :]), torch.Tensor(locations[:, 1:, :])
        node_demand = torch.Tensor(demand[1:, 1:].reshape((1, -1))) / capacity  # [1, n]

        env_params = {'problem_size': node_xy.size(1), 'pomo_size': node_xy.size(1), 'loc_scaler': loc_scaler, 'device': self.device}
        env = Env(**env_params)
        data = (depot_xy, node_xy, node_demand)
        _, _, no_aug_score, aug_score = self._test_one_batch(model, env, data)
        no_aug_score = torch.round(no_aug_score * loc_scaler).long()
        aug_score = torch.round(aug_score * loc_scaler).long()
        print(">> Finish solving {} -> no_aug: {} aug: {}".format(path, no_aug_score, aug_score))

        return no_aug_score, aug_score
