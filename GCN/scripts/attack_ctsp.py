import numpy as np
from tsp.data import GoogleTSPReader
from tsp.utils import *
from tsp.attacks import attack_rand, attack_opt
import argparse
from tsp.data import get_CTSP_training_data
from tsp.solvers.ctsp import get_convtsp_model


def val_tsp(net, val, pert="random", batch_size=2, num_neighbors=20, test_robustness=True):
    print(">> {} attacking...".format(pert))
    net.eval()
    dataset = GoogleTSPReader(num_nodes=100, num_neighbors=num_neighbors, batch_size=batch_size, filepath=val)  # TODO: num_neighbors = -1
    batches_per_epoch = dataset.max_iter  # 10000 // 20 = 500
    dataset = iter(dataset)
    running_loss, running_pred_tour_len, running_gt_tour_len, running_gap = 0.0, 0.0, 0.0, 0.0
    running_nb_data, running_nb_batch = 0, 0
    losses, preds, gts, gaps = [], [], [], []

    for batch_num in range(batches_per_epoch):
        try:
            batch = next(dataset)
        except StopIteration:
            break

        if test_robustness:
            x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, y_nodes = unroll(batch)
            batch_attack = {
                'coords': [x.cpu() for x in x_nodes_coord],
                'routes': [y.tolist() for y in y_nodes],
                'og_batch': batch
            }
            if pert == "random":
                outs = attack_rand(batch_attack, net, num_neighbors)
            elif pert == "opt":
                outs = attack_opt(batch_attack, net, num_neighbors)
            else:
                raise NotImplementedError()
            y_preds, loss = net.forward(outs)
            pred_tour_len, gt_tour_len = padded_get_stats(y_preds, outs)
        else:
            y_preds, loss = net.forward(batch)
            pred_tour_len, gt_tour_len = padded_get_stats(y_preds, batch)

        print(pred_tour_len)
        print(gt_tour_len)
        losses.extend(loss.tolist())
        preds.extend(pred_tour_len.tolist())
        gts.extend(gt_tour_len.tolist())
        gap = np.array([(pred_tour_len[i].item()-gt_tour_len[i].item())/gt_tour_len[i].item()*100 for i in range(batch_size)])
        gaps.extend(gap.tolist())
        print("Batch {}/500, Loss: {}, pred_len: {}, gt_len: {}, Gap: {}".format(batch_num+1, np.array(losses).mean(), np.array(preds).mean(), np.array(gts).mean(), np.array(gaps).mean()))
        
        running_nb_data += batch_size
        running_loss += batch_size * loss.mean().data.item()
        running_pred_tour_len += batch_size * pred_tour_len.mean()
        running_gt_tour_len += batch_size * gt_tour_len.mean()
        running_gap += batch_size * gap.mean()
        running_nb_batch += 1

    loss = running_loss / running_nb_data
    pred_length = running_pred_tour_len / running_nb_data
    gt_length = running_gt_tour_len / running_nb_data
    opt_gap = running_gap / running_nb_data

    print(">> Val Finished: Loss: {}, pred_len: {}, gt_len: {}, Gap: {}".format(loss, pred_length, gt_length, opt_gap))
    
    return loss, pred_length, gt_length, opt_gap


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run Attack')
    parser.add_argument('-seed', default=0, action='store_true', help='Python random seed')
    parser.add_argument('-opt', default=False, action='store_true', help='Optimized or random attack')
    # parser.add_argument('-model_path', default="../trained_models/trained_ctsp.pt", action='store_true', help='Model to be attacked')
    parser.add_argument('-model_path', default="../trained_models/ctsp100_last_train_checkpoint.tar", action='store_true', help='Model to be attacked')
    args = parser.parse_args()

    model, _ = get_convtsp_model(args.seed)
    model.cuda()
    params = torch.load(args.model_path)
    params_new = dict()
    if len(params.keys()) != 369:
        params = params['model_state_dict']
    for key in params.keys():
        params_new[key[7:]] = params[key]
    model.load_state_dict(params_new)
    train, val, test, name = get_CTSP_training_data()
    res = val_tsp(model, val, pert="opt" if args.opt else "random")
