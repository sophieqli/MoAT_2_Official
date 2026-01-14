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

import copy

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

def avg_ll(model, loader, alpha=None, loader_contracted=None):
    lls = []
    dataset_len = 0
    sig_a = torch.sigmoid(model.alpha) if alpha is not None else None

    def forward_mix(x_batch, use_orig=False):
        """helper: forward pass through mixture of moats"""
        logits = []
        for k in range(model.K):
            logits.append(
                model(
                    x_batch,
                    which_moat=k,
                    W=model.W_orig if use_orig else None,
                    V_compress=model.V_orig if use_orig else None,
                    E=model.E_orig if use_orig else None,
                ) + torch.log(mix_ws[k])
            )
        return torch.logsumexp(torch.stack(logits, dim=0), dim=0)

    for batch in (zip(loader, loader_contracted) if loader_contracted is not None else loader):
        # handle both dataset cases
        if loader_contracted is not None:
            x_batch_orig, x_batch_con = batch
            x_batch_orig = x_batch_orig.to(device)
            x_batch_con = x_batch_con.to(device)
        else:
            x_batch_orig = batch.to(device)
            x_batch_con = None

        # normalize mixture weights
        mix_ws = torch.exp(model.mix_ws)
        mix_ws = mix_ws / mix_ws.sum()

        # forward passes
        y_batch_orig = forward_mix(x_batch_orig, use_orig=(alpha is not None))
        if sig_a is not None and x_batch_con is not None:
            y_batch_con = forward_mix(x_batch_con)
            y_batch = torch.log(sig_a * torch.exp(y_batch_con) + (1 - sig_a) * torch.exp(y_batch_orig))
        else:
            y_batch = y_batch_orig

        # accumulate
        ll = torch.sum(y_batch)
        lls.append(ll.item())
        dataset_len += x_batch_orig.shape[0]

    return torch.sum(torch.tensor(lls)).item() / dataset_len

def ipfp_update(V, E, K):
    """Perform IPFP updates on edge potentials E given marginals V."""
    EPS = torch.tensor(1e-7)
    with torch.no_grad():
        V_soft = torch.softmax(V, dim=1)
        for which_moat in range(K):
            E_proj = torch.exp(E[which_moat])
            E_proj = E_proj / (E_proj.sum(dim=(-2, -1), keepdim=True) + EPS)

            A_b = V_soft[:, None, :, None]   # [n, 1, l, 1]
            B_b = V_soft[None, :, None, :]   # [1, n, 1, l]

            for _ in range(200):
                row_marg = E_proj.sum(dim=3, keepdim=True)
                E_proj.mul_(A_b / (row_marg + EPS))
                col_marg = E_proj.sum(dim=2, keepdim=True)
                E_proj.mul_(B_b / (col_marg + EPS))
                E_proj /= (E_proj.sum(dim=(-2, -1), keepdim=True) + EPS)

            E[which_moat].copy_(torch.log(E_proj + EPS))
        # log back the V
        V.copy_(torch.log(V_soft + EPS))

def train_model(
    model, train, valid, test, lr, weight_decay, batch_size, max_epoch, log_file, output_model_file, dataset_name, alpha=None,
    train_contracted=None,
    valid_contracted=None,
    test_contracted=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === create static loaders for validation/test (unchanged) ===
    if valid is not None:
        valid_loader = DataLoader(dataset=valid, batch_size=batch_size, shuffle=False)
    if test is not None:
        test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

    # contracted presence flag
    have_contracted = train_contracted is not None
    if have_contracted:
        assert len(train) == len(train_contracted), "orig/contracted lengths must match EXACTLY"

    # === STATIC contracted loaders for val / test ===
    if have_contracted:
        valid_loader_contracted = (
            DataLoader(valid_contracted, batch_size=batch_size, shuffle=False)
            if valid_contracted is not None else None
        )
        test_loader_contracted = (
            DataLoader(test_contracted, batch_size=batch_size, shuffle=False)
            if test_contracted is not None else None
        )
    else:
        valid_loader_contracted = None
        test_loader_contracted = None


    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model = model.to(device)
    max_valid_ll = -1.0e7
    model.train()
    best_val = -1000000
    best_test = 10
    best_train = 10
    best_val_epoch = 0

    print("max epoch is ", max_epoch)

    val_log = [] 
    # === training loop ===
    for epoch in range(max_epoch):
        print(f"Epoch {epoch}")
        step_cnt = 0

        # Create ONE shared random permutation for this epoch (guaranteed alignment)
        perm = torch.randperm(len(train))

        # Apply the same permutation to both datasets
        train_perm = torch.utils.data.Subset(train, perm)
        if have_contracted:
            train_con_perm = torch.utils.data.Subset(train_contracted, perm)

        # Create per-epoch loaders WITHOUT shuffle (they're already randomized)
        # Use drop_last=True so both loaders have identical number of batches
        train_loader = DataLoader(train_perm, batch_size=batch_size, shuffle=False, drop_last=True)
        if have_contracted:
            train_loader_contracted = DataLoader(train_con_perm, batch_size=batch_size, shuffle=False, drop_last=True)
        else: 
            train_loader_contracted = None
            valid_loader_contracted = None
            test_loader_contracted = None


        # zip loaders if we have both datasets
        if have_contracted:
            loader = zip(train_loader, train_loader_contracted)
        else:
            loader = train_loader

        for batch_idx, batch in enumerate(loader):
            # handle both dataset cases
            if train_loader_contracted is not None:
                x_batch_orig, x_batch_contracted = batch
                x_batch_orig = x_batch_orig.to(device)
                x_batch_contracted = x_batch_contracted.to(device)
                #print("x_batch_orig.shape ", x_batch_orig.shape)
                #print("x_batch_contracted.shape ", x_batch_contracted.shape)
            else:
                x_batch_orig = batch.to(device)
                x_batch_contracted = None

            mix_ws = torch.exp(model.mix_ws)
            mix_ws = mix_ws / mix_ws.sum()

            # --- forward on original dataset ---
            logits_orig = []
            for k in range(model.K):
                if alpha is not None: 
                    logits_orig.append(
                        model(x_batch_orig, epoch=epoch, which_moat=k, step_cnt=step_cnt, W = model.W_orig, V_compress = model.V_orig, E=model.E_orig)
                        + torch.log(mix_ws[k])
                    )
                else: 
                    logits_orig.append(
                        model(x_batch_orig, epoch=epoch, which_moat=k, step_cnt=step_cnt)
                        + torch.log(mix_ws[k]))


            y_batch_orig = torch.logsumexp(torch.stack(logits_orig, dim=0), dim=0)

            # --- forward on contracted dataset if available ---
            if alpha is not None and x_batch_contracted is not None:
                logits_con = []
                for k in range(model.K):
                    ll_con = model(
                        x_batch_contracted, epoch=epoch, which_moat=k, step_cnt=step_cnt) + torch.log(mix_ws[k])
                    logits_con.append(ll_con)

                y_batch_con = torch.logsumexp(torch.stack(logits_con, dim=0), dim=0)

                # blend them in log-space
                y_batch = torch.log(
                    torch.sigmoid(model.alpha)  * torch.exp(y_batch_con) + (1 - torch.sigmoid(model.alpha)) * torch.exp(y_batch_orig)
                )
                #print("orig y batch orig ", torch.exp(y_batch_orig), "; contracted loss is ", torch.exp(y_batch_con))
                #print("weights are ", torch.sigmoid(model.alpha), " and ", (1 - torch.sigmoid(model.alpha)))
            else:
                y_batch = y_batch_orig


            # === compute loss ===
            loss = nll(y_batch, model.E_compress, model.V_compress)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            step_cnt += 1

            # --- normalize contracted parameters (the active model) ---
            ipfp_update(model.V_compress, model.E_compress, model.K)

            # --- normalize deleted/original parameters if present ---
            if hasattr(model, "V_orig") and hasattr(model, "E_orig"):
                ipfp_update(model.V_orig, model.E_orig, model.K)

        if alpha is not None:
            # === Marginal Consistency Diagnostics ===
            with torch.no_grad():
                V_prob = torch.softmax(model.V_compress, dim=-1)
                E_prob = torch.exp(model.E_compress[0])
                E_prob = E_prob / E_prob.sum(dim=(-2, -1), keepdim=True)
                marg_consistency_diagnostics(V_prob, E_prob)


        # compute likelihood on train, valid and test
        if alpha == None: 
            train_ll = avg_ll(model, train_loader)
            valid_ll = avg_ll(model, valid_loader)
            test_ll = avg_ll(model, test_loader)
        else: 
            train_ll = avg_ll(model, train_loader, alpha=model.alpha, loader_contracted=train_loader_contracted)
            valid_ll = avg_ll(model, valid_loader, alpha=model.alpha, loader_contracted=valid_loader_contracted)
            test_ll  = avg_ll(model, test_loader,  alpha=model.alpha, loader_contracted=test_loader_contracted)

        if valid_ll > best_val:
            best_val_epoch = epoch
            best_train, best_val, best_test = train_ll, valid_ll, test_ll

        print('Dataset {}; Epoch {}; train ll: {}; valid ll: {}; test ll: {}'.format(dataset_name, epoch, train_ll, valid_ll, test_ll))

        val_log.append(valid_ll)
        #Check for consecutive increasing likelihoods
        if len(val_log) > 3:
            if val_log[-1] < val_log[-2] < val_log[-3] < val_log[-4]:
                print("Breaking out of training loop - beginning to overfit")
                break

        wandb.log({"train_ll": train_ll, "val_ll": test_ll}, step=epoch)
        with open(log_file, 'a+') as f:
            f.write('{} {} {} {}\n'.format(epoch, train_ll, valid_ll, test_ll))
        if output_model_file != '' and valid_ll > max_valid_ll:
            torch.save(model, output_model_file)
            max_valid_ll = valid_ll

    #out of loop, print final stats
    print("Best_val_epoch: ", best_val_epoch, " best_val: ", best_val, "best_test: ", best_test, "best_train", best_train) 

def merge_columns_and_append(x: torch.Tensor, xi: int, xj: int) -> torch.Tensor:
    """
    Merge two columns (xi, xj) of tensor x into a new single column y = 2*xi + xj,
    then return tensor with xi and xj columns removed and y appended at the end.

    Args:
        x (torch.Tensor): Input tensor of shape (num_samples, num_features)
        xi (int): Index of first column to merge
        xj (int): Index of second column to merge

    Returns:
        torch.Tensor: Modified tensor with merged column appended, shape (num_samples, num_features - 1)
    """
    mask = torch.ones(x.shape[1], dtype=torch.bool)
    mask[xi] = False
    mask[xj] = False

    x_reduced = x[:, mask]
    x_i = x[:, xi]
    x_j = x[:, xj]
    y = 2 * x_i + x_j  # merge columns
    y = y.unsqueeze(1)

    #return with i,j cols and without removed 
    return torch.cat([x, y], dim=1), torch.cat([x_reduced, y], dim=1)

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
        tags=["imagenet", "4by4patches", "4-bits"],
        mode="offline"   # <--- add this here

    )

    #gives us the dataset files
    train, valid, test = load_data2(args.dataset_path, args.dataset)

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
        model = MoAT(n=2, x=t_data, num_classes=4, device='cpu')
        model.to(device)
        train_loader = DataLoader(dataset=train, batch_size=args.batch_size, shuffle=True)
        print('average ll: {}'.format(avg_ll(model, train_loader)))

    if model is None:
        print("invalid model")
        exit(1)

    #initial training, none compressed
    train_model(model, train=train, valid=valid, test=test,
        lr=args.lr, weight_decay=args.weight_decay,
        batch_size=args.batch_size, max_epoch=args.max_epoch,
        log_file=args.log_file, output_model_file=args.output_model_file,
        dataset_name=args.dataset)

    #Split into G\e and G-e
    #xi, xj = 1, 3
    sub_mat = model.edge_posts()
    max_val = sub_mat.max()
    max_pos = sub_mat.argmax()
    #xi, xj = max_pos // model.n, max_pos % model.n
    xi, xj = torch.div(max_pos, model.n, rounding_mode='trunc'), max_pos % model.n
    print("max posterior edge: ", xi, xj, " -> probability: ", max_val)

    train_0 = copy.deepcopy(train)
    valid_0 = copy.deepcopy(valid) if valid is not None else None
    test_0  = copy.deepcopy(test)  if test is not None else None

    _, train.x = merge_columns_and_append(train.x, xi, xj)
    _, valid.x = merge_columns_and_append(valid.x, xi, xj) if valid is not None else None
    _, test.x = merge_columns_and_append(test.x, xi, xj) if test is not None else None
    model.contract_and_delete(xi, xj, train.x, post_prob = max_val.item())
    print("After merge_columns on contracted:")
    print("train_0.x shape:", train_0.x.shape, "(orig)")
    print("train.x shape:", train.x.shape, "(contracted)")
    
    #now split param training 
    train_model(
        model, train=train_0, valid=valid_0, test=test_0, lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, max_epoch=args.max_epoch, log_file=args.log_file, output_model_file=args.output_model_file, dataset_name=args.dataset, alpha=model.alpha,
        train_contracted=train,
        valid_contracted=valid,
        test_contracted=test)


    # Binary compression:
    #IGNORE FOR NOW - MI grouping not used
    num_merges = 0 #4 achieves better LL, nltcs!!!
    simul = 2

    for merge_idx in range(num_merges):
        for i in range(simul):
            n_bin = model.n - model.b4
            sub_mat = model.edge_posts()[:n_bin, :n_bin]
            max_val = sub_mat.max()
            max_pos = sub_mat.argmax()
            xi, xj = max_pos // n_bin, max_pos % n_bin
            '''
            #min posterior logic 
            edge_posts = model.edge_posts()
            mask = torch.ones_like(edge_posts, dtype=torch.bool)
            mask.fill_diagonal_(False)  # diagonal = False, off-diagonal = True

            # Masked values: set diagonal to some large number so they are never minimum
            masked_edge_posts = edge_posts.clone()
            masked_edge_posts[~mask] = float('inf')  # or a large positive number

            # Find min and argmin among off-diagonal elements
            sub_mat = masked_edge_posts[:n_bin, :n_bin]
            min_val = sub_mat.min()
            min_pos = sub_mat.argmin()
            xi = min_pos // n_bin
            xj = min_pos % n_bin
            '''
            #print(f"Max gradient edge at ({xi}, {xj}) with value {max_val.item()}")
            print("modifying train dataset ")
            _, train.x = merge_columns_and_append(train.x, xi, xj)
            _, valid.x = merge_columns_and_append(valid.x, xi, xj) if valid is not None else None
            _, test.x = merge_columns_and_append(test.x, xi, xj) if test is not None else None

            model.contract_edge_b2tob4_params(xi, xj, train.x)
            print(" performing merge #: ", merge_idx)

        train_model(model, train=train, valid=valid, test=test,
            lr=args.lr, weight_decay=args.weight_decay,
            batch_size=args.batch_size, max_epoch=args.max_epoch,
            log_file=args.log_file, output_model_file=args.output_model_file,
            dataset_name=args.dataset)

    wandb.finish()

if __name__ == '__main__':
    main()
