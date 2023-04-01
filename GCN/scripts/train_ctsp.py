import torch
import numpy as np
from datetime import datetime
from tsp.data import GoogleTSPReader
from tsp.utils import *
import argparse
from tsp.data import get_CTSP_training_data
from tsp.solvers.ctsp import get_convtsp_model


def train_convtsp(train, val, dataset_name, net, learning_rate=0.001,
                  batches_per_epoch=500, decay_rate=1.01, epochs=1700, batch_size=20):
    net.cuda()
    net.train()
    val_loss_old, best_pred_tour_len = 1e6, 1e6 
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    losses, pred_lengths, gt_lengths = [], [], []
    val_losses, val_pred_lengths = [], []
    simulation_name = 'ConvTSP_' + dataset_name + '_' + datetime.now().strftime("%d:%m-%H:%M:%S")
    print(simulation_name)
    
    for _ in range(epochs):
        dataset = GoogleTSPReader(20, -1, batch_size, train)
        dataset = iter(dataset)
        running_loss, running_pred_tour_len, running_gt_tour_len = 0.0, 0.0, 0.0
        running_nb_data, running_nb_batch = 0, 0

        for _ in range(batches_per_epoch):
            try:
                batch = next(dataset)
            except StopIteration:
                break

            y_preds, loss = net.forward(batch)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            pred_tour_len, gt_tour_len = padded_get_stats(y_preds, batch)

            # Update running data
            running_nb_data += batch_size
            running_loss += batch_size* loss.data.item()
            running_pred_tour_len += batch_size* pred_tour_len.mean().item()
            running_gt_tour_len += batch_size* gt_tour_len.mean().item()
            running_nb_batch += 1

        # Compute statistics for full epoch
        losses.append(running_loss/ running_nb_data)
        pred_lengths.append(running_pred_tour_len/ running_nb_data)
        gt_lengths.append(running_gt_tour_len/ running_nb_data)
        print("Train loss", (running_loss/ running_nb_data))

        val_loss, val_pred_tour_len, _ = val_tsp(net, val)
        print("Val loss", val_loss)
        val_losses.append(val_loss)
        val_pred_lengths.append(val_pred_tour_len)

        if val_pred_tour_len < best_pred_tour_len:
            torch.save(net.state_dict(), f'../trained_models/{simulation_name}_MAX.pt')
            best_pred_tour_len = val_pred_tour_len

        if val_loss > 0.99 * val_loss_old:
            learning_rate /= decay_rate
            optimizer = update_learning_rate(optimizer, learning_rate)        
        val_loss_old = val_loss  

    return_dict = {
        'Name': simulation_name,
        'Train Loss': losses,
        'Train Pred Len': pred_lengths,
        'Train GT Len': gt_lengths,
        'Val Loss': val_losses,
        'Val Pred Len': val_pred_lengths
    }
    return return_dict


def val_tsp(net, val, batch_size=15):
    net.eval()
    dataset = GoogleTSPReader(20, -1, batch_size, val)
    batches_per_epoch = dataset.max_iter
    dataset = iter(dataset)
    running_loss, running_pred_tour_len, running_gt_tour_len = 0.0, 0.0, 0.0
    running_nb_data, running_nb_batch = 0, 0
    losses, preds, gts = [], [], []

    for batch_num in range(batches_per_epoch):
        try:
            batch = next(dataset)
        except StopIteration:
            break
        
        y_preds, loss = net.forward(batch)

        pred_tour_len, gt_tour_len = padded_get_stats(y_preds, batch)
        losses.extend(loss.tolist())
        preds.extend(pred_tour_len.tolist())
        gts.extend(gt_tour_len.tolist())
        
        running_nb_data += batch_size
        running_loss += batch_size * loss.mean().data.item()
        running_pred_tour_len += batch_size * pred_tour_len.mean()
        running_gt_tour_len += batch_size * gt_tour_len.mean()
        running_nb_batch += 1

    loss = running_loss / running_nb_data
    pred_length = running_pred_tour_len / running_nb_data
    gt_length = running_gt_tour_len / running_nb_data
     
    return loss, pred_length, gt_length 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run training')
    args = parser.parse_args()

    model, _ = get_convtsp_model(0)
    train, val, test, name = get_CTSP_training_data()
    res = train_convtsp(train, val, name, model)
    print(res)



