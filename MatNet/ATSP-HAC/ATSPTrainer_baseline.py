import os, random
import glob
import torch
from logging import getLogger

from ATSPEnv import ATSPEnv as Env
from ATSPModel import ATSPModel as Model
from ATSProblemDef import get_random_problems, load_single_problem_from_file, generate_x_adv
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from utils.utils import *
from utils.functions import load_dataset, save_dataset


class ATSPTrainer:
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

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        pretrain_load = trainer_params['pretrain_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{n}-{epoch}.pt'.format(**model_load)
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
            self.logger.info('Checkpoint loaded successfully from {}'.format(checkpoint_fullname))

        elif pretrain_load['enable']:  # (Only) Load pretrain model
            checkpoint_fullname = '{path}/checkpoint-{n}-{epoch}.pt'.format(**pretrain_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            for i in range(self.num_expert):
                self.models[i].load_state_dict(checkpoint['model_state_dict'])
            self.logger.info('Pretrain model loaded successfully from {}'.format(checkpoint_fullname))

        else:  # pretrain (phase 1) from scratch
            self.logger.info('No pretrain model found! Pretraining from scratch.')
            for epoch in range(self.trainer_params['pretrain_epochs']+1):
                self._train_one_epoch(epoch, mode="nat")
            model_state_dict = self.pre_model.state_dict()
            for i in range(self.num_expert):
                self.models[i].load_state_dict(model_state_dict)
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
            # Note: needs to regenerate adv instances using generate_x_adv
            dirs = ["../../data/ATSP-HAC/test_n20", ]
            val_episodes, score_list, gap_list = 1000, [], []
            for dir in dirs:
                score, gap = self._val_and_stat(dir, batch_size=100, val_episodes=val_episodes)
                score_list.append(score);
                gap_list.append(gap)
            self.result_log.append('val_score', epoch, score_list)
            self.result_log.append('val_gap', epoch, gap_list)

            # Logs & Checkpoint
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            # img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch > 1:  # save latest images, every epoch
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'], self.result_log, labels=['val_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'], self.result_log, labels=['val_gap'])

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': [model.state_dict() for model in self.models],
                    'optimizer_state_dict': [optimizer.state_dict() for optimizer in self.optimizers],
                    'scheduler_state_dict': [scheduler.state_dict() for scheduler in self.schedulers],
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if all_done:
                self.logger.info(" *** Training Done *** ")
                # self.logger.info("Now, printing log array...")
                # util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch, mode="nat"):
        """
            Baseline Method.
            Phase 1 mode == "nat":
                Pre-training one model on natural instances.
            Phase 2 mode == "adv":
                One pretrain model -> several experts
                    - Generating adversarial instances;
                    - Train on (nat+)adv ins. generated by itself
        """
        episode = 0
        train_num_episode = self.trainer_params['train_episodes']

        while episode < train_num_episode:
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)
            nat_data = get_random_problems(batch_size, self.env_params["node_cnt"], self.env_params["problem_gen_params"])

            if mode == "nat":
                score, loss = self._train_one_batch(self.pre_model, nat_data)
                avg_score, avg_loss = score.mean().item(), loss.mean()
                self.pre_optimizer.zero_grad()
                avg_loss.backward()
                self.pre_optimizer.step()
            elif mode == "adv":
                # generate adv_data
                # Note: Could further add scheduler to control attack budget (e.g., curriculum way).
                eps = random.sample(range(self.adv_params['eps_min'], self.adv_params['eps_max']+1), 1)[0]
                for i in range(self.num_expert):
                    adv_data = generate_x_adv(self.models[i], nat_data, eps=eps, num_steps=self.adv_params['num_steps'], return_opt=False)

                    if self.trainer_params['method'] == "baseline":
                        score, loss = self._train_one_batch(self.models[i], adv_data)  # (batch), (batch)
                        avg_score, avg_loss = score.mean().item(), loss.mean()
                    elif self.trainer_params['method'] == "baseline_hac":
                        """
                            Reimplementation of HAC, refer to "Learning to Solve Travelling Salesman Problem with Hardness-Adaptive Curriculum" - AAAI 2022
                            https://github.com/wondergo2017/TSP-HAC
                        """
                        hac_data = torch.cat((nat_data, adv_data), dim=0)  # nat+adv
                        hac_data = hac_data[torch.randperm(hac_data.size(0))[:batch_size]]
                        avg_score, avg_loss = self._hac_train_one_batch(self.models[i], hac_data, epoch=epoch)
                    else:
                        raise NotImplementedError

                    self.optimizers[i].zero_grad()
                    avg_loss.backward()
                    self.optimizers[i].step()
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
        self.env.load_problems_manual(data)
        reset_state, _, _ = self.env.reset()
        model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~)

        # POMO Rollout
        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss & Score
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        # loss_mean = loss.mean()
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        # score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        return -max_pomo_reward.float().detach(), loss.mean(1)  # (batch), (batch)

    def _hac_train_one_batch(self, model, data, epoch=0):
        """
            Only for one forward pass.
        """
        model.train()
        batch_size = data.size(0)
        self.env.load_problems_manual(data)
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

        # Score
        ###############################################
        max_pomo_reward, idx = reward.max(dim=1)  # get best results from pomo
        # score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Reweighting by HAC
        loss = loss.mean(1)
        w = ((max_pomo_reward / reward.float().mean(dim=1)) * log_prob[torch.arange(batch_size), idx]).detach()
        t = torch.FloatTensor([20 - (epoch % 20)]).to(self.device)
        w = torch.tanh(w) / t
        w = torch.softmax(w, dim=0)
        loss = (w * loss).sum()

        return -max_pomo_reward.float().detach().mean().item(), loss  # (batch), (batch)

    def _fast_val(self, model, data=None, path=None, offset=0, val_episodes=1000, aug_factor=1, eval_type="softmax"):
        data = torch.Tensor(load_dataset(path, disable_print=True)[offset: offset + val_episodes]) if data is None else data
        data = data.to(self.device)
        env = Env(**{'node_cnt': data.size(1), 'problem_gen_params': {'int_min': 0, 'int_max': 1000 * 1000, 'scaler': 1000 * 1000}, 'pomo_size': data.size(1), 'device': self.device})
        batch_size = aug_factor * data.size(0)
        data = data.repeat(aug_factor, 1, 1)

        model.eval()
        model.set_eval_type(eval_type)
        with torch.no_grad():
            env.load_problems_manual(data)
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)
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
        no_aug_score = -max_pomo_reward[0, :].float()  # negative sign to make positive value
        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        aug_score = -max_aug_pomo_reward.float()  # negative sign to make positive value

        return no_aug_score, aug_score

    def _val_and_stat(self, dir, batch_size=500, val_episodes=1000):
        no_aug_score_list, aug_score_list, no_aug_gap_list, aug_gap_list = [], [], [], []
        no_aug_scores, aug_scores = torch.zeros(val_episodes, 0), torch.zeros(val_episodes, 0)

        val_data = torch.zeros(0, self.env_params['node_cnt'], self.env_params['node_cnt'])
        opt_sol = load_dataset("{}.pkl".format(dir), disable_print=True)[: val_episodes]
        opt_sol = [i[0] for i in opt_sol]
        for fp in sorted(glob.iglob(os.path.join(dir, "*.atsp"))):
            data = load_single_problem_from_file(fp, node_cnt=self.env_params['node_cnt'], scaler=1000 * 1000)
            val_data = torch.cat((val_data, data.unsqueeze(0)), dim=0)
        val_data = val_data[: val_episodes].to(self.device)

        for i in range(self.num_expert):
            episode, no_aug_score, aug_score = 0, torch.zeros(0).to(self.device), torch.zeros(0).to(self.device)
            while episode < val_episodes:
                remaining = val_episodes - episode
                bs = min(batch_size, remaining)
                no_aug, aug = self._fast_val(self.models[i], data=val_data[episode: episode+bs], aug_factor=16, eval_type="softmax")
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
            dir, no_aug_score_list, min(no_aug_score_list), moe_no_aug_score, aug_score_list, min(aug_score_list), moe_aug_score))
        print(">> Val Score on {}: NO_AUG_Gap: {} -> Min {}% -> Col {}%, AUG_Gap: {} -> Min {}% -> Col {}%".format(
            dir, no_aug_gap_list, min(no_aug_gap_list), moe_no_aug_gap, aug_gap_list, min(aug_gap_list), moe_aug_gap))

        return moe_aug_score, moe_aug_gap
