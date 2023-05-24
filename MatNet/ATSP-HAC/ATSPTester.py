import torch
import os, time, glob
from logging import getLogger

from ATSPEnv import ATSPEnv as Env
from ATSPModel import ATSPModel as Model
from utils.utils import get_result_folder, AverageMeter, TimeEstimator
from ATSProblemDef import load_single_problem_from_file
from utils.functions import load_dataset, save_dataset


class ATSPTester:
    def __init__(self, env_params, model_params, tester_params, model=None):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

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

        if model is not None:
            self.env = Env(**self.env_params)
            self.model = model.to(self.device)
        else:
            # result folder, logger
            self.logger = getLogger(name='trainer')
            self.result_folder = get_result_folder()

            # ENV and MODEL
            self.num_expert = self.tester_params['num_expert']
            self.env = Env(**self.env_params)
            if self.num_expert == 1:
                self.model = Model(**self.model_params)
            else:
                self.models = [Model(**self.model_params) for _ in range(self.num_expert)]

            # load dataset
            self.test_data = torch.zeros(0, self.env_params['node_cnt'], self.env_params['node_cnt'])
            for fp in sorted(glob.iglob(os.path.join(self.tester_params['saved_problem_folder'], "*.atsp"))):
                data = load_single_problem_from_file(fp, node_cnt=self.env_params['node_cnt'], scaler=1000 * 1000)
                self.test_data = torch.cat((self.test_data, data.unsqueeze(0)), dim=0)
            opt_sol = load_dataset(self.tester_params['test_set_opt_sol_path'], disable_print=True)[: self.tester_params['file_count']]  # [(obj, route), ...]
            print(self.test_data.size())
            self.test_data = self.test_data[: self.tester_params['file_count']].to(self.device)
            self.opt_sol = [i[0] for i in opt_sol]

            # Restore
            model_load = tester_params['model_load']
            checkpoint_fullname = '{path}/checkpoint-{n}-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            if self.num_expert == 1:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.models = [self.model]
            else:
                model_state_dict = checkpoint['model_state_dict']
                for i in range(self.num_expert):
                    self.models[i].load_state_dict(model_state_dict[i])

            # utility
            self.time_estimator = TimeEstimator()

            # Load all problems into tensor
            self.logger.info(" *** Loading Saved Problems *** ")
            saved_problem_folder = self.tester_params['saved_problem_folder']
            saved_problem_filename = self.tester_params['saved_problem_filename']
            file_count = self.tester_params['file_count']
            node_cnt = self.env_params['node_cnt']
            scaler = self.env_params['problem_gen_params']['scaler']
            self.all_problems = torch.empty(size=(file_count, node_cnt, node_cnt))
            for file_idx in range(file_count):
                formatted_filename = saved_problem_filename.format(file_idx)
                full_filename = os.path.join(saved_problem_folder, formatted_filename)
                problem = load_single_problem_from_file(full_filename, node_cnt, scaler)
                self.all_problems[file_idx] = problem
            self.logger.info("Done. ")

    def run(self, i):
        start_time = time.time()
        scores, aug_scores = self._test(self.models[i])
        print(">> Evaluation on {} finished within {:.2f}s".format(self.tester_params['saved_problem_folder'], time.time() - start_time))

        return scores.cpu(), aug_scores.cpu(), self.opt_sol, i

    def _test(self, model):
        self.time_estimator.reset()
        score_AM, gap_AM = AverageMeter(), AverageMeter()
        aug_score_AM, aug_gap_AM = AverageMeter(), AverageMeter()
        scores, aug_scores = torch.zeros(0).to(self.device), torch.zeros(0).to(self.device)

        test_num_episode = self.tester_params['file_count']
        assert len(self.test_data) == test_num_episode, "the number of test instances does not match!"
        episode = 0
        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)
            score, aug_score, all_score, all_aug_score = self._test_one_batch(model, self.env, self.test_data[episode: episode + batch_size])
            opt_sol = self.opt_sol[episode: episode + batch_size]

            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)
            episode += batch_size
            gap = [(all_score[i].item() - opt_sol[i]) / opt_sol[i] * 100 for i in range(batch_size)]
            aug_gap = [(all_aug_score[i].item() - opt_sol[i]) / opt_sol[i] * 100 for i in range(batch_size)]
            gap_AM.update(sum(gap)/batch_size, batch_size)
            aug_gap_AM.update(sum(aug_gap)/batch_size, batch_size)
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
        batch_size = test_data.size(0)
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
            batch_size = aug_factor * batch_size
            test_data = test_data.repeat(aug_factor, 1, 1)
        else:
            aug_factor = 1

        # Ready
        model.eval()
        model.set_eval_type(self.model_params["eval_type"])
        with torch.no_grad():
            env.load_problems_manual(problems=test_data)
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)

        # POMO Rollout
        state, reward, done = env.pre_step()
        while not done:
            selected, _ = model(state)
            # shape: (batch, pomo)
            state, reward, done = env.step(selected)

        # Return
        batch_size = batch_size // aug_factor
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
