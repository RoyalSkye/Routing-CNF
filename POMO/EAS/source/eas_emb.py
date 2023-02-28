import math
import time

import numpy as np
import torch.optim as optim
from tqdm import tqdm
from source.TORCH_OBJECTS import *
AUG_S = 8  # Number of augmentations


def run_eas_emb(model, instance_data, problem_size, config, get_episode_data_fn, augment_and_repeat_episode_data_fn):
    """
    Efficient active search using embedding updates
    """
    dataset_size = len(instance_data[0])  # np.array

    assert config.batch_size <= dataset_size

    instance_solutions = torch.zeros(dataset_size, problem_size * 2, dtype=torch.int)
    instance_costs = np.zeros((dataset_size))

    if config.problem == "TSP":
        from source.tsp.env import GROUP_ENVIRONMENT
    elif config.problem == "CVRP":
        from source.cvrp.env import GROUP_ENVIRONMENT

    for episode in range(math.ceil(dataset_size / config.batch_size)):

        print(">> {}: {}/{} instances finished.".format(config.method, episode * config.batch_size, dataset_size))
        episode_data = get_episode_data_fn(instance_data, episode * config.batch_size, config.batch_size, problem_size)
        batch_size = episode_data[0].shape[0]  # Number of instances considered in this iteration

        p_runs = config.p_runs  # Number of parallel runs per instance
        batch_r = batch_size * p_runs  # Search runs per batch
        batch_s = AUG_S * batch_r  # Model batch size (nb. of instances * the number of augmentations * p_runs)
        group_s = problem_size + 1  # Number of different rollouts per instance (+1 for incumbent solution construction)

        with torch.no_grad():
            aug_data = augment_and_repeat_episode_data_fn(episode_data, problem_size, p_runs, AUG_S)
            env = GROUP_ENVIRONMENT(aug_data, problem_size, config.round_distances, loc_scaler=config.loc_scaler)
            group_state, reward, done = env.reset(group_size=group_s)
            # model.reset(group_state)  # Generate the embeddings (i.e., k, v, and single_head_key)
            model.pre_forward(group_state)

        # We do not update all embeddings, but only the single single_head_key that has been generated by the encoder. All model weights are held fixed.
        model.requires_grad_(False)
        model.decoder.single_head_key.requires_grad_(True)
        # model.node_prob_calculator.single_head_key.requires_grad_(True)
        optimizer = optim.Adam([model.decoder.single_head_key], lr=config.param_lr, weight_decay=1e-6)

        incumbent_solutions = torch.zeros(batch_size, problem_size * 2, dtype=torch.int)

        # Start the search
        ###############################################
        t_start = time.time()
        for iter in range(config.max_iter):
            group_state, reward, done = env.reset(group_size=group_s)
            incumbent_solutions_expanded = incumbent_solutions.repeat(AUG_S, 1).repeat(p_runs, 1)

            # Start generating batch_s * group_s solutions
            ###############################################
            solutions = []
            step = 0
            if config.problem == "CVRP":
                # First Move is given
                first_action = LongTensor(np.zeros((batch_s, group_s)))  # start from node_0-depot
                # model(group_state, selected=first_action)  # do nothing for CVRP
                group_state, reward, done = env.step(first_action)
                solutions.append(first_action.unsqueeze(2))
                step += 1

            # First/Second Move is given
            second_action = LongTensor(np.arange(group_s) % problem_size)[None, :].expand(batch_s, group_s).clone()
            if iter > 0:
                second_action[:, -1] = incumbent_solutions_expanded[:, step]  # Teacher forcing imitation learning loss
            model(group_state, selected=second_action)  # for the first step, set_q1 for TSP, do nothing for CVRP
            group_state, reward, done = env.step(second_action)
            solutions.append(second_action.unsqueeze(2))
            step += 1

            group_prob_list = Tensor(np.zeros((batch_s, group_s, 0)))
            while not done:
                action_probs = model(group_state)
                # action_probs = model.get_action_probabilities(group_state)  # shape = (batch_s, group_s, problem)
                action = action_probs.reshape(batch_s * group_s, -1).multinomial(1).squeeze(dim=1).reshape(batch_s, group_s)  # shape = (batch_s, group_s)
                if iter > 0:
                    action[:, -1] = incumbent_solutions_expanded[:, step]  # Teacher forcing the imitation learning loss

                if config.problem == "CVRP":
                    action[group_state.finished] = 0  # stay at depot, if you are finished
                group_state, reward, done = env.step(action)
                solutions.append(action.unsqueeze(2))

                batch_idx_mat = torch.arange(int(batch_s))[:, None].expand(batch_s, group_s)
                group_idx_mat = torch.arange(group_s)[None, :].expand(batch_s, group_s)
                chosen_action_prob = action_probs[batch_idx_mat, group_idx_mat, action].reshape(batch_s, group_s)
                # shape = (batch_s, group_s)
                if config.problem == "CVRP":
                    chosen_action_prob[group_state.finished] = 1  # done episode will gain no more probability
                group_prob_list = torch.cat((group_prob_list, chosen_action_prob[:, :, None]), dim=2)
                step += 1

            # Solution generation finished. Update incumbent solutions and best rewards
            ###############################################
            group_reward = reward.reshape(AUG_S, batch_r, group_s)
            solutions = torch.cat(solutions, dim=2)
            if config.batch_size == 1:
                # Single instance search. Only a single incumbent solution exists that needs to be updated
                max_idx = torch.argmax(reward)
                best_solution_iter = solutions.reshape(-1, solutions.shape[2])
                best_solution_iter = best_solution_iter[max_idx]
                incumbent_solutions[0, :best_solution_iter.shape[0]] = best_solution_iter
                max_reward = reward.max()
            else:
                # Batch search. Update incumbent etc. separately for each instance
                max_reward, _ = group_reward.max(dim=2)
                max_reward, _ = max_reward.max(dim=0)
                reward_g = group_reward.permute(1, 0, 2).reshape(batch_r, -1)
                iter_max_k, iter_best_k = torch.topk(reward_g, k=1, dim=1)
                solutions = solutions.reshape(AUG_S, batch_r, group_s, -1)
                solutions = solutions.permute(1, 0, 2, 3).reshape(batch_r, AUG_S * group_s, -1)
                best_solutions_iter = torch.gather(solutions, 1, iter_best_k.unsqueeze(2).expand(-1, -1, solutions.shape[2])).squeeze(1)
                incumbent_solutions[:, :best_solutions_iter.shape[1]] = best_solutions_iter

            # LEARNING - Actor
            # Use the same reinforcement learning method as during the training of the model to update the
            # embeddings node_prob_calculator.single_head_key
            ###############################################
            group_reward = reward[:, :group_s - 1]
            # shape = (batch_s, group_s - 1)
            group_log_prob = group_prob_list.log().sum(dim=2)
            # shape = (batch_s, group_s)
            group_advantage = group_reward - group_reward.mean(dim=1, keepdim=True)

            group_loss = -group_advantage * group_log_prob[:, :group_s - 1]
            # shape = (batch_s, group - 1)
            loss_1 = group_loss.mean()  # Reinforcement learning loss
            loss_2 = -group_log_prob[:, group_s - 1].mean()  # Imitation learning loss
            loss = loss_1 + loss_2 * config.param_lambda

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if time.time() - t_start > config.max_runtime:
                break

        # Store incumbent solutions and their objective function value
        instance_solutions[episode * config.batch_size: episode * config.batch_size + batch_size] = incumbent_solutions
        instance_costs[episode * config.batch_size: episode * config.batch_size + batch_size] = -max_reward.cpu().numpy()

    return instance_costs, instance_solutions
