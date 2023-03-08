import os, random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model
from TSPModel import TSP_Routing
from TSProblemDef import get_random_problems, generate_x_adv
from TSP_baseline import solve_concorde_log
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from generate_adv import generate_adv_dataset
from utils.utils import *
from utils.functions import *


class TSPTrainer:
    def __init__(self, env_params, model_params, optimizer_params, trainer_params, adv_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        self.adv_params = adv_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device
        self.env_params['device'] = device
        self.model_params['device'] = device

        # Main Components
        self.num_expert = self.trainer_params['num_expert']
        self.env = Env(**self.env_params)
        # pretraining for phase 1
        self.pre_model = Model(**self.model_params)
        self.pre_optimizer = Optimizer(self.pre_model.parameters(), **self.optimizer_params['optimizer'])
        # several experts for phase 2
        self.models = [Model(**self.model_params) for _ in range(self.num_expert)]
        self.optimizers = [Optimizer(model.parameters(), **self.optimizer_params['optimizer']) for model in self.models]
        self.schedulers = [Scheduler(optimizer, **self.optimizer_params['scheduler']) for optimizer in self.optimizers]
        self.routing_model = TSP_Routing(num_expert=self.num_expert) if self.trainer_params['routing_model'] else None
        self.routing_optimizer = Optimizer(self.routing_model.parameters(), **self.optimizer_params['optimizer']) if self.routing_model else None
        # self.routing_optimizer = torch.optim.SGD(self.routing_model.parameters(), lr=0.01) if self.routing_model else None

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        pretrain_load = trainer_params['pretrain_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            model_state_dict = checkpoint['model_state_dict']
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            for i in range(self.num_expert):
                model, optimizer, scheduler = self.models[i], self.optimizers[i], self.schedulers[i]
                model.load_state_dict(model_state_dict[i])
                optimizer.load_state_dict(optimizer_state_dict[i])
                scheduler.last_epoch = model_load['epoch'] - 1
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            if self.routing_model:
                self.routing_model.load_state_dict(checkpoint['routing_model_state_dict'])
                self.routing_optimizer.load_state_dict(checkpoint['routing_optimizer_state_dict'])
            self.logger.info('Checkpoint loaded successfully from {}'.format(checkpoint_fullname))

        elif pretrain_load['enable']:  # (Only) Load pretrain model
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**pretrain_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            # self.pre_model.load_state_dict(checkpoint['model_state_dict'])
            # TODO: Only load Encoder?
            for i in range(self.num_expert):
                model = self.models[i]
                model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info('Pretrain model loaded successfully from {}'.format(checkpoint_fullname))

        else:  # pretrain (phase 1) from scratch
            self.logger.info('No pretrain model found! Pretraining from scratch.')
            for epoch in range(self.trainer_params['pretrain_epochs']+1):
                self._train_one_epoch(epoch, mode="nat")
            model_state_dict = self.pre_model.state_dict()
            for i in range(self.num_expert):
                model = self.models[i]
                model.load_state_dict(model_state_dict)
            del self.pre_model
            self.logger.info('Pretraining finished.')

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # Train
            self._train_one_epoch(epoch, mode="adv")
            for i in range(self.num_expert):
                self.schedulers[i].step()

            # Validation
            dir = "../../data/TSP"
            paths = ["tsp100_uniform.pkl", "adv_tsp100_uniform.pkl"]
            val_episodes, score_list, gap_list = 1000, [], []
            nat_data = torch.Tensor(load_dataset(os.path.join(dir, paths[0]), disable_print=True)[: val_episodes]).to(self.device)
            # generate adv dataset based on the status of current model
            self._generate_cur_adv(nat_data)

            for path in paths:
                score, gap = self._val_and_stat(dir, path, batch_size=500, val_episodes=val_episodes)
                score_list.append(score); gap_list.append(gap)
            score, gap = self._val_and_stat("./", "adv_tmp.pkl", batch_size=500, val_episodes=val_episodes * self.num_expert)
            score_list.append(score); gap_list.append(gap)
            self.result_log.append('val_score', epoch, score_list)
            self.result_log.append('val_gap', epoch, gap_list)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            # img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch > 1:  # save latest images, every epoch
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                # util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'], self.result_log, labels=['train_score'])
                # util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'], self.result_log, labels=['train_loss'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'], self.result_log, labels=['val_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'], self.result_log, labels=['val_gap'])

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': [model.state_dict() for model in self.models],
                    'optimizer_state_dict': [optimizer.state_dict() for optimizer in self.optimizers],
                    'scheduler_state_dict': [scheduler.state_dict() for scheduler in self.schedulers],
                    'routing_model_state_dict': self.routing_model.state_dict() if self.routing_model else None,
                    'routing_optimizer_state_dict': self.routing_optimizer.state_dict() if self.routing_model else None,
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if all_done:
                self.logger.info(" *** Training Done *** ")
                # self.logger.info("Now, printing log array...")
                # util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch, mode="nat"):
        """
            Improve the Robustness of Neural Heuristics through Experts Collaboration.
            Phase 1 mode == "nat":
                Pre-training one model on natural instances.
            Phase 2 mode == "adv":
                One pretrain model -> several experts
                    - Generating adversarial instances;
                    - Train on nat+adv ins. with routing mechanism
        """
        # score_AM, loss_AM = AverageMeter(), AverageMeter()
        episode = 0
        train_num_episode = self.trainer_params['train_episodes']

        while episode < train_num_episode:
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)
            nat_data = get_random_problems(batch_size, self.env_params['problem_size'])

            if mode == "nat":
                # forward pass
                score, loss = self._train_one_batch(self.pre_model, nat_data)
                avg_score, avg_loss = score.mean().item(), loss.mean()
                # backward pass
                self.pre_optimizer.zero_grad()
                avg_loss.backward()
                self.pre_optimizer.step()
            elif mode == "adv":
                eps = random.sample(range(self.adv_params['eps_min'], self.adv_params['eps_max']+1), 1)[0]
                # eps = 1 + int(1 / 2 * (1 - math.cos(math.pi * min(epoch / 30, 1))) * (100 - 1))  # cosine

                # 1. generate adversarial examples by each expert (local)
                for i in range(self.num_expert):
                    adv_data = generate_x_adv(self.models[i], nat_data, eps=eps, num_steps=self.adv_params['num_steps'], return_opt=False)
                    data = torch.cat((nat_data, adv_data), dim=0)  # nat+adv
                    scores = torch.zeros(batch_size * 2, 0)
                    for j in range(self.num_expert):
                        _, score = self._fast_val(self.models[j], data=data, aug_factor=1, eval_type="softmax")
                        scores = torch.cat((scores, score.unsqueeze(1)), dim=1)
                    # print(scores)  # the scores will not be the same even at the beginning, since the policy is stochastic
                    if self.routing_model:
                        self._update_model_routing(data, scores, type="exp_choice")
                    else:
                        self._update_model(data, scores, type="exp_choice")

                # 2. collaborate to generate adversarial examples (global)
                data = nat_data
                for _ in range(self.adv_params['num_steps']):
                    adv_data, scores = torch.zeros(0, data.size(1), 2), torch.zeros(batch_size, 0)
                    for k in range(self.num_expert):
                        _, score = self._fast_val(self.models[k], data=data, aug_factor=1, eval_type="softmax")
                        scores = torch.cat((scores, score.unsqueeze(1)), dim=1)
                    _, id = scores.min(1)
                    for k in range(self.num_expert):
                        mask = (id == k)
                        if mask.sum() < 1: continue
                        data_ = generate_x_adv(self.models[k], data[mask], eps=eps, num_steps=1, return_opt=False)
                        adv_data = torch.cat((adv_data, data_), dim=0)
                    data = adv_data
                data = torch.cat((nat_data, data), dim=0)  # nat+adv
                scores = torch.zeros(batch_size * 2, 0)
                for k in range(self.num_expert):
                    _, score = self._fast_val(self.models[k], data=data, aug_factor=1, eval_type="softmax")
                    scores = torch.cat((scores, score.unsqueeze(1)), dim=1)
                if self.routing_model:
                    self._update_model_routing(data, scores, type="exp_choice")
                else:
                    self._update_model(data, scores, type="exp_choice")

            else:
                raise NotImplementedError

            episode += batch_size

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)'.format(epoch, 100. * episode / train_num_episode))

    def _train_one_batch(self, model, data):
        """
            Only for one forward pass.
        """
        model.train()
        batch_size = data.size(0)
        self.env.load_problems(batch_size, problems=data, aug_factor=1)
        reset_state, _, _ = self.env.reset()
        model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        ###############################################
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        # loss_mean = loss.mean()

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        # score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        return -max_pomo_reward.float().detach(), loss.mean(1)  # (batch), (batch)

    def _update_model(self, data, scores, type="ins_choice"):
        """
            Updating model using both nat_data and adv_data.
            Routing Mechanism:
                a. Each instance chooses its best expert (cons: no load_balance)
                b. Each expert chooses TopK instances based on relative gaps (cons: some instances may not be trained)
                c. Combine together
                d. jointly train a routing network (see self._update_model_routing)
        """
        batch_size = scores.size(0)
        if type == "ins_choice":
            _, id = scores.min(1)  # (batch) - instance choice
        elif type == "exp_choice":
            gaps = (scores - scores.min(1)[0].view(-1, 1)) / scores.min(1)[0].view(-1, 1)  # relative gaps among all experts
            _, id = gaps.topk(batch_size // 2, dim=0, largest=False)  # (k, num_expert) - expert choice
        elif type == "ins_exp_choice":
            gaps = (scores - scores.min(1)[0].view(-1, 1)) / scores.min(1)[0].view(-1, 1)
            _, id1 = gaps.min(1)
            _, id2 = gaps.topk(batch_size // 2, dim=0, largest=False)

        for j in range(self.num_expert):
            if type == "ins_choice":
                mask = (id == j)
            elif type == "exp_choice":
                mask = id[:, j]
            elif type == "ins_exp_choice":
                mask1 = (id1 == j)
                mask2 = torch.zeros(batch_size).bool().scatter_(0, id2[:, j], 1)
                mask = mask1 | mask2
            elif type == "normal":
                mask = torch.ones(batch_size).bool()
            else:
                raise NotImplementedError
            selected_data = data[mask]
            if selected_data.size(0) < 1: continue
            _, loss = self._train_one_batch(self.models[j], selected_data)
            avg_loss = loss.mean()
            self.optimizers[j].zero_grad()
            avg_loss.backward()
            self.optimizers[j].step()

    def _update_model_routing(self, data, scores, type="ins_choice", alpha=0.1, temp=1.0):
        """
            Updating model using both nat_data and adv_data, which are routed by a routing network.
                - jointly train a routing network (see MOE - Mixture-Of-Experts)
        """
        # TODO: optimizer (Adam/SGD)? sample/topk? temp
        state = scores
        # state = scores / scores.max(1)[0].unsqueeze(1)
        # state = (scores - scores.min(1)[0].view(-1, 1)) / scores.min(1)[0].view(-1, 1)
        logits = self.routing_model(data, state) / temp  # (batch_size, num_expert)
        batch_size, routing_loss, avg_scores = scores.size(0), torch.zeros(0), torch.mean(scores, dim=1)

        if type == "ins_choice":
            probs = torch.softmax(logits, dim=1)
            # id = torch.multinomial(probs, num_samples=1, replacement=False).squeeze(1)  # (batch_size)
            id = torch.topk(probs, 1, dim=1, largest=True, sorted=False)[1].squeeze(1)
            # print(probs)
        elif type == "exp_choice":
            probs = torch.softmax(logits, dim=0)
            # id = torch.multinomial(probs.T, num_samples=batch_size//2, replacement=False).T  # (batch_size//2, num_expert)
            id = torch.topk(probs, batch_size//2, dim=0, largest=True, sorted=False)[1]
            # print(probs)
        elif type == "ins_exp_choice":
            probs1, probs2 = torch.softmax(logits, dim=1), torch.softmax(logits, dim=0)
            # id1 = torch.multinomial(probs1, num_samples=1, replacement=False).squeeze(1)
            # id2 = torch.multinomial(probs2.T, num_samples=batch_size//2, replacement=False).T
            id1 = torch.topk(probs1, 1, dim=1, largest=True, sorted=False)[1].squeeze(1)
            id2 = torch.topk(probs2, batch_size//2, dim=0, largest=True, sorted=False)[1]
            # print(torch.cat((probs1, probs2), dim=1))

        # routing and training
        for j in range(self.num_expert):
            if type == "ins_choice":
                mask = (id == j)
            elif type == "exp_choice":
                mask = id[:, j]
            elif type == "ins_exp_choice":
                mask1, mask2 = (id1 == j), torch.zeros(batch_size).bool().scatter_(0, id2[:, j], 1)
                mask = mask1 | mask2
            else:
                raise NotImplementedError
            selected_data = data[mask]
            if selected_data.size(0) < 1: continue
            score, loss = self._train_one_batch(self.models[j], selected_data)
            avg_loss = loss.mean()
            self.optimizers[j].zero_grad()
            avg_loss.backward()
            self.optimizers[j].step()

        new_scores = torch.zeros(batch_size, 0)
        for j in range(self.num_expert):
            _, score = self._fast_val(self.models[j], data=data, aug_factor=1, eval_type="softmax")
            new_scores = torch.cat((new_scores, score.unsqueeze(1)), dim=1)

        # update routing network
        reward = (scores.min(1, keepdim=True)[0] - new_scores.min(1, keepdim=True)[0]).expand(-1, self.num_expert)  # (batch_size, num_expert)
        if type == "ins_choice":
            log_prob = torch.log(probs)
            reward, log_prob = torch.gather(reward, 1, id.unsqueeze(1)), torch.gather(log_prob, 1, id.unsqueeze(1))
            routing_loss = (-reward * log_prob).mean()
        elif type == "exp_choice":
            log_prob = torch.log(probs)
            reward, log_prob = torch.gather(reward, 0, id), torch.gather(log_prob, 0, id)
            routing_loss = (-reward * log_prob).mean()
        elif type == "ins_exp_choice":
            log_prob1, log_prob2 = torch.log(probs1), torch.log(probs2)
            reward1, reward2 = torch.gather(reward, 1, id1.unsqueeze(1)), torch.gather(reward, 0, id2)
            log_prob1, log_prob2 = torch.gather(log_prob1, 1, id1.unsqueeze(1)), torch.gather(log_prob2, 0, id2)
            routing_loss = alpha * (-reward1 * log_prob1).mean() + (1 - alpha) * (-reward2 * log_prob2).mean()
        else:
            raise NotImplementedError

        # update routing network
        self.routing_optimizer.zero_grad()
        routing_loss.backward()
        self.routing_optimizer.step()

    def _fast_val(self, model, data=None, path=None, offset=0, val_episodes=1000, aug_factor=1, eval_type="argmax"):
        data = torch.Tensor(load_dataset(path, disable_print=True)[offset: offset + val_episodes]) if data is None else data
        data = data.to(self.device)
        env = Env(**{'problem_size': data.size(1), 'pomo_size': data.size(1)})
        batch_size = data.size(0)

        model.eval()
        model.set_eval_type(eval_type)
        with torch.no_grad():
            env.load_problems(batch_size, problems=data, aug_factor=aug_factor)
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)
            state, reward, done = env.pre_step()
            while not done:
                selected, _ = model(state)
                # shape: (batch, pomo)
                state, reward, done = env.step(selected)

        # Return
        aug_reward = reward.reshape(aug_factor, batch_size, env.pomo_size)
        # shape: (augmentation, batch, pomo)
        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        no_aug_score = -max_pomo_reward[0, :].float()  # negative sign to make positive value
        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        aug_score = -max_aug_pomo_reward.float()  # negative sign to make positive value

        return no_aug_score, aug_score

    def _generate_cur_adv(self, nat_data):
        # generate adv examples based on current models
        adv_data = torch.zeros(0, self.env_params['problem_size'], 2).to(self.device)
        for j in range(self.num_expert):
            data = generate_adv_dataset(self.models[j], nat_data, eps_min=self.adv_params['eps_min'], eps_max=self.adv_params['eps_max'], num_steps=self.adv_params['num_steps'])
            adv_data = torch.cat((adv_data, data), dim=0)
        save_dataset(adv_data, "./adv_tmp.pkl")

        # obtain (sub-)opt solution using Concorde
        params = argparse.ArgumentParser()
        params.cpus, params.n, params.progress_bar_mininterval = None, None, 0.1
        dataset = [(instance.cpu().numpy(),) for instance in adv_data]
        executable = os.path.abspath(os.path.join('concorde', 'concorde', 'TSP', 'concorde'))
        def run_func(args):
            return solve_concorde_log(executable, *args, disable_cache=True)
        results, _ = run_all_in_pool(run_func, "./Concorde_result", dataset, params, use_multiprocessing=False)
        os.system("rm -rf ./Concorde_result")
        results = [(i[0], i[1]) for i in results]
        save_dataset(results, "./concorde_adv_tmp.pkl")

    def _val_and_stat(self, dir, val_path, batch_size=500, val_episodes=1000):
        no_aug_score_list, aug_score_list, no_aug_gap_list, aug_gap_list = [], [], [], []
        no_aug_scores, aug_scores = torch.zeros(val_episodes, 0), torch.zeros(val_episodes, 0)
        opt_sol = load_dataset(os.path.join(dir, "concorde_{}".format(val_path)), disable_print=True)[: val_episodes]
        opt_sol = [i[0] for i in opt_sol]
        for i in range(self.num_expert):
            episode, no_aug_score, aug_score = 0, torch.zeros(0).to(self.device), torch.zeros(0).to(self.device)
            while episode < val_episodes:
                remaining = val_episodes - episode
                bs = min(batch_size, remaining)
                no_aug, aug = self._fast_val(self.models[i], path=os.path.join(dir, val_path), offset=episode, val_episodes=bs, aug_factor=8, eval_type="argmax")
                no_aug_score = torch.cat((no_aug_score, no_aug), dim=0)
                aug_score = torch.cat((aug_score, aug), dim=0)
                episode += bs

            no_aug_score_list.append(round(no_aug_score.mean().item(), 4))
            aug_score_list.append(round(aug_score.mean().item(), 4))
            no_aug_scores = torch.cat((no_aug_scores, no_aug_score.unsqueeze(1)), dim=1)
            aug_scores = torch.cat((aug_scores, aug_score.unsqueeze(1)), dim=1)
            gap = [(no_aug_score[j].item() - opt_sol[j]) / opt_sol[j] * 100 for j in range(val_episodes)]
            no_aug_gap_list.append(round(sum(gap) / len(gap), 4))
            gap = [(aug_score[j].item() - opt_sol[j]) / opt_sol[j] * 100 for j in range(val_episodes)]
            aug_gap_list.append(round(sum(gap) / len(gap), 4))
        moe_no_aug_score, moe_aug_score = no_aug_scores.min(1)[0], aug_scores.min(1)[0]
        moe_no_aug_gap = [(moe_no_aug_score[j].item() - opt_sol[j]) / opt_sol[j] * 100 for j in range(val_episodes)]
        moe_aug_gap = [(moe_aug_score[j].item() - opt_sol[j]) / opt_sol[j] * 100 for j in range(val_episodes)]
        moe_no_aug_gap, moe_aug_gap = sum(moe_no_aug_gap) / len(moe_no_aug_gap), sum(moe_aug_gap) / len(moe_aug_gap)
        moe_no_aug_score, moe_aug_score = moe_no_aug_score.mean().item(), moe_aug_score.mean().item()

        print(">> Val Score on {}: NO_AUG_Score: {} -> Min {} Col {}, AUG_Score: {} -> Min {} -> Col {}".format(
            val_path, no_aug_score_list, min(no_aug_score_list), moe_no_aug_score, aug_score_list, min(aug_score_list), moe_aug_score))
        print(">> Val Score on {}: NO_AUG_Gap: {} -> Min {}% -> Col {}%, AUG_Gap: {} -> Min {}% -> Col {}%".format(
            val_path, no_aug_gap_list, min(no_aug_gap_list), moe_no_aug_gap, aug_gap_list, min(aug_gap_list), moe_aug_gap))

        return moe_aug_score, moe_aug_gap
