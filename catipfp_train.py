import os
os.environ["WANDB_API_KEY"] = "8b8c0680523ebb6d5faab719618395d412cec4ef"

import wandb

import random
import numpy as np
import torch
# Make sure operations are deterministic (optional, slows training a tiny bit)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import numpy.linalg as la
from sklearn.cluster import SpectralClustering
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import json


import os
import math
import argparse

from catipfp_model import *
from cont_copula import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class DatasetFromFile(Dataset):
    def __init__(self, filename):
        examples = []
        with open(filename, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line == '':
                    continue
                line = [int(x) for x in line.split(',')]
                examples.append(line)
        x = torch.LongTensor(examples)
        self.x = x

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return len(self.x)

def init():
    global device
    global CUDA_CORE

    torch.set_default_dtype(torch.float64)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset_path', default='', type=str)
    arg_parser.add_argument('--dataset', default='', type=str)
    arg_parser.add_argument('--device', default='cuda', type=str)
    arg_parser.add_argument('--cuda_core', default='0', type=str)
    arg_parser.add_argument('--model', default='DPP', type=str)
    arg_parser.add_argument('--max_epoch', default=20, type=int)
    arg_parser.add_argument('--batch_size', default=8, type=int)
    arg_parser.add_argument('--lr', default=0.001, type=float)
    arg_parser.add_argument('--weight_decay', default=0.0, type=float)
    arg_parser.add_argument('--component_num', default=10, type=int)
    arg_parser.add_argument('--max_cluster_size', default=10, type=int)
    arg_parser.add_argument('--log_file', default='log.txt', type=str)
    arg_parser.add_argument('--output_model_file', default='model.pt', type=str)
    arg_parser.add_argument('--evidence_idx', default=0, type=int)

    args = arg_parser.parse_args()

    device = args.device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_core

    return args

def load_data2(dataset_path, dataset,
            load_train=True, load_valid=True, load_test=True):
    dataset_path += '{}/'.format(dataset)
    train_path = dataset_path + '{}.train.data'.format(dataset)
    valid_path = dataset_path + '{}.valid.data'.format(dataset)
    test_path = dataset_path + '{}.test.data'.format(dataset)

    train, valid, test = None, None, None

    if load_train:
        train = DatasetFromFile(train_path)
    if load_valid:
        valid = DatasetFromFile(valid_path)
    if load_test:
        test = DatasetFromFile(test_path)

    return train, valid, test

def partition_variables(trainx, max_cluster_size):
    n = len(trainx)
    m = len(trainx[0])
    k = max_cluster_size

    freq = {}
    for i in range(0, m):
        freq[i] = 0
        for j in range(i + 1, m):
            freq[(i, j)] = 0
    for t in tqdm(range(0, n)):
        for i in range(0, m):
            if trainx[t][i] == 1:
                freq[i] += 1
                for j in range(i + 1, m):
                    if trainx[t][j] == 1:
                        freq[(i, j)] += 1
    for i in freq:
        freq[i] /= n

    E = []
    for i in range(0, m):
        if abs(freq[i]) < 1e-15:
            continue
        for j in range(i + 1, m):
            if abs(freq[j]) < 1e-15:
                continue
            p = freq[(i, j)] / (freq[i] * freq[j])
            if p < 1.0:
                continue
            w = freq[(i, j)] * math.log(p)
            E.append(((i, j), w))
    E = sorted(E, key=lambda x: x[1], reverse=True)

    fa = [i for i in range(0, m)]
    def find(x):
        if fa[x] == x:
            return x
        fa[x] = find(fa[x])
        return fa[x]

    def count(x):
        cnt = 0
        for i in range(0, m):
            if find(i) == x:
                cnt += 1
        return cnt

    set_cnt = m
    for e, w in E:
        if w < 0:
            break
        u, v = e
        fu, fv = find(u), find(v)
        size_u, size_v = count(fu), count(fv)
        if size_u + size_v > k:
            continue
        fa[u] = fv
        fa[fu] = fv
        if fu != fv:
            set_cnt -= 1

    for i in range(0, m):
        fa[i] = find(i)

    res = {}
    for u in range(0, m):
        fu = fa[u]
        if fu not in res:
            res[fu] = []
        res[fu].append(u)

    partition = []
    for k, v in res.items():
        partition.append(v)

    return partition

def nll(y, E_compress, V):
    ll = -torch.sum(y)
    return ll

def avg_ll(model, dataset_loader):
    lls = []
    dataset_len = 0
    for x_batch in dataset_loader:
        x_batch = x_batch.to(device)
        
        #now y_batch is a mixture, rather than y_batch = model(x_batch)
        y_batch = torch.zeros(x_batch.shape[0], device = x_batch.device)

        mix_ws = torch.exp(model.mix_ws)
        mix_ws = mix_ws / mix_ws.sum() #normalize weights to add to 1
        logits = []
        for _ in range(model.K):
            logits.append(model(x_batch, which_moat=_) + torch.log(mix_ws[_]))
        y_batch = torch.logsumexp(torch.stack(logits, dim=0), dim=0)

        ll = torch.sum(y_batch)
        lls.append(ll.item())
        dataset_len += x_batch.shape[0]

    avg_ll = torch.sum(torch.Tensor(lls)).item() / dataset_len

    return avg_ll

def train_model(model, train, valid, test,
                lr, weight_decay, batch_size, max_epoch,
                log_file, output_model_file, dataset_name):
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=False) #CHANGE BACK TO TRUE
    if valid is not None:
        valid_loader = DataLoader(dataset=valid, batch_size=batch_size, shuffle=True)
    if test is not None:
        test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # training loop
    max_valid_ll = -1.0e7
    model = model.to(device)
    model.train()
    best_val = -1000000
    best_test = 10
    best_train = 10
    best_val_epoch = 0

    for epoch in range(0, 51): 
        print('Epoch: {}'.format(epoch))

        step_cnt = 0 
        data_for_epoch = [] 
        # step in train
        for x_batch in train_loader:
            entry = {}

            x_batch = x_batch.to(device)
            #ybatch has shape x_batch.shape[0]
            y_batch = torch.zeros(x_batch.shape[0], device = x_batch.device)
            
            mix_ws = torch.exp(model.mix_ws)
            mix_ws = mix_ws / mix_ws.sum() #normalize weights to add to 1
            logits = []
            for _ in range(model.K):
                logits.append(model(x_batch, epoch=epoch, which_moat=_, step_cnt = step_cnt) + torch.log(mix_ws[_]))
            y_batch = torch.logsumexp(torch.stack(logits, dim=0), dim=0)

            loss = nll(y_batch, model.E_compress, model.V_compress)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # clip gradients after loss.backward()
            #entry['x'] = torch.exp(model.E_compress[0])[1,2,0,0].item()
            #entry['y'] = torch.exp(model.E_compress[0])[1,2,0,1].item()

            optimizer.step()
            step_cnt += 1
            #try ipfp here to project original parameters -- qccumulate gradient 

            EPS = torch.tensor(1e-7, device=model.V_compress.device)

            with torch.no_grad():
                model.V_compress.copy_(torch.softmax(model.V_compress, dim=1))
                for which_moat in range(model.K):
                    # Project E_compress[which_moat] into positive space and normalize
                    E_proj = torch.exp(model.E_compress[which_moat])
                    E_proj = E_proj / (E_proj.sum(dim=(-2, -1), keepdim=True) + EPS)

                    #entry['x_step'] = E_proj[1,2,0,0].item()
                    #entry['y_step'] = E_proj[1,2,0,1].item()

                    E_step = E_proj.clone()

                    A_b = model.V_compress[:, None, :, None]  # [n, 1, l, 1]
                    B_b = model.V_compress[None, :, None, :]  # [1, n, 1, l]

                    for _ in range(100):
                        row_marg = E_proj.sum(dim=3, keepdim=True) 
                        E_proj.mul_(A_b / row_marg)
                        col_marg = E_proj.sum(dim=2, keepdim=True)
                        E_proj.mul_(B_b / col_marg)
                        E_proj /= E_proj.sum(dim=(-2, -1), keepdim=True) + EPS 

                    #Take the KLDF here 
                    #entry['x_proj'] = E_proj[1,2,0,0].item()
                    #entry['y_proj'] = E_proj[1,2,0,1].item()
                    model.E_compress[which_moat].copy_(torch.log(E_proj + EPS))
                # Also log-transform V_compress again
                model.V_compress.copy_(torch.log(model.V_compress + EPS))

            data_for_epoch.append(entry)
        ''' 
        with open("trajectory_log1.jsonl", "a") as f:
            f.write(json.dumps(data_for_epoch) + "\n")
        '''      
        # compute likelihood on train, valid and test
        train_ll = avg_ll(model, train_loader)
        valid_ll = avg_ll(model, valid_loader)
        test_ll = avg_ll(model, test_loader)
        if valid_ll > best_val:
            best_val = valid_ll
            best_val_epoch = epoch
            best_test = test_ll
            best_train = train_ll

        print('Dataset {}; Epoch {}; train ll: {}; valid ll: {}; test ll: {}'.format(dataset_name, epoch, train_ll, valid_ll, test_ll))

        wandb.log({"train_ll": train_ll, "val_ll": test_ll}, step=epoch)

        with torch.no_grad():
            print("=== V_compress Stats ===")
            V_compress, E = torch.softmax(model.V_compress, dim = -1),torch.exp(model.E_compress) #convert back to raw probabilities
            row_max = V_compress.max(dim=1).values
            row_entropy = -(V_compress* V_compress.log()).sum(dim=1)
            row_mean = V_compress.mean(dim=1).values
            E = E / E.sum(dim=(-2, -1), keepdim=True)
            joint_entropy = -(E * E.clamp(min=1e-10).log()).sum(dim=(-2, -1))

            # === Marginal Consistency Diagnostics ===
            row_margs = E.sum(dim=-1)  # [K, n, n, l]
            col_margs = E.sum(dim=-2)  # [K, n, n, l]
            V_i = V_compress[None, :, None, :]  # [1, n, 1, l]
            V_j = V_compress[None, None, :, :]  # [1, 1, n, l]

            row_diff = torch.abs(row_margs - V_i)
            col_diff = torch.abs(col_margs - V_j)
            #print("sahpes of row and col diff shld be K x n x n x l: ", row_margs.shape, col_margs.shape)
            diag_mask = 1 - torch.eye(model.n, device=E.device).unsqueeze(-1).unsqueeze(0)  # [1, n, n, 1] ?? 
            row_diff = row_diff * diag_mask
            col_diff = col_diff * diag_mask

            row_mae = row_diff.mean().item()
            col_mae = col_diff.mean().item()
            row_median = row_diff.flatten().median().item()
            col_median = col_diff.flatten().median().item()
            max_row = row_diff.max().item()
            max_col = col_diff.max().item()

            print("\n=== Marginal Consistency Check ===")
            print(f"Mean abs row marginal diff (E vs V): {row_mae:.6f}")
            print(f"Mean abs col marginal diff (E vs V): {col_mae:.6f}")
            print(f"Median abs row marginal diff (E vs V): {row_median:.6f}")
            print(f"Median abs col marginal diff (E vs V): {col_median:.6f}")
            print(f"Max row marginal diff: {max_row:.6f}")
            print(f"Max col marginal diff: {max_col:.6f}")

        with open(log_file, 'a+') as f:
            f.write('{} {} {} {}\n'.format(epoch, train_ll, valid_ll, test_ll))

        if output_model_file != '' and valid_ll > max_valid_ll:
            torch.save(model, output_model_file)
            max_valid_ll = valid_ll

    #out of loop, print final distribution
    print("Best_val_epoch: ", best_val_epoch, " best_val: ", best_val, "best_test: ", best_test, "best_train", best_train) 



def main():
    args = init()

    wandb.login()

    config = {
        "epochs": args.max_epoch,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dataset": args.dataset,
    }

    wandb.init(
        project="MoATexps",
        config=config,
        name=f"run_{args.dataset}",
        tags=["imagenet", "4by4patches", "4-bits"]
    )

    #gives us the dataset files
    train, valid, test = load_data2(args.dataset_path, args.dataset)

    #t, v, train_load, val_load = load_data("/scratch/sophie_li/data" )

    print('train: {}'.format(train.x.shape))
    if valid is not None:
        print('valid: {}'.format(valid.x.shape))
    if test is not None:
        print('test: {}'.format(test.x.shape))

    m = train.x.shape[1]

    model = None

    if args.model == 'MoAT':
        t_data=train.x.clone()
        t_data.to(device)
        model = MoAT(n=2, x=t_data, num_classes=2, device='cpu')
        model.to(device)
        train_loader = DataLoader(dataset=train, batch_size=args.batch_size, shuffle=True)
        print('average ll: {}'.format(avg_ll(model, train_loader)))
        #print('average ll: {}'.format(avg_ll(model, train_load)))

    if model is None:
        print("invalid model")
        exit(1)

    train_model(model, train=train, valid=valid, test=test,
        lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch_size, max_epoch=args.max_epoch,
        log_file=args.log_file, output_model_file=args.output_model_file,
        dataset_name=args.dataset)
    
    wandb.finish()


if __name__ == '__main__':
    main()
