import os, sys
import argparse, tsplib95
from attacker_env import AttackerEnv
from ATSP_algorithms import get_adj
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for utils
from utils.functions import load_dataset, save_dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default='lkh-5', choices=["nn", "furthest", "lkh-5"])
    parser.add_argument('-node_dim', type=int, default=20, help="Problem sizes of instances")
    parser.add_argument("--datasets", nargs='+', default=["./data/test_n20", "./data/test_adv_n20", ], help="Filename of the dataset(s) to evaluate")
    parser.add_argument("--opt", nargs='+', default=["./data/test_n20.pkl", "./data/test_adv_n20.pkl", ], help="Filename of the optimal solution")
    parser.add_argument('-n', type=int, default=1000, help="Number of instances to process")
    parser.add_argument('--save', action='store_true', help='Whether save results or not')

    args = parser.parse_args()

    for i, dataset_path in enumerate(args.datasets):
        atsp_env = AttackerEnv(solver_type=args.method, node_dimension=args.node_dim, is_attack=True, tester=None, path=dataset_path)
        total_cost, total_gap, total_time = 0, 0, 0
        opt, results = None, []
        if len(args.opt) != 0:
            opt = load_dataset(args.opt[i], disable_print=True)[: args.n]
            opt = [i[0] for i in opt]
        for i in range(args.n):
            tsp_path = atsp_env.tspfiles[i]
            print(tsp_path)
            problem = tsplib95.load(tsp_path)
            lower_left_matrix = get_adj(problem)
            tour, sol, sec = atsp_env.solve_feasible_tsp(lower_left_matrix, args.method)
            total_time += sec
            total_cost += sol
            results.append([sol, tour])
            if opt is not None:
                total_gap += (sol-opt[i])/opt[i] * 100

        print(">> {} - {} - AVG_Cost: {} AVG_Gap: {}% within {}s".format(args.method, dataset_path, total_cost/args.n, total_gap/args.n, total_time/args.n))

        if args.save:
            save_dataset(results, "{}.pkl".format(dataset_path))  # [(obj, route), ...]
