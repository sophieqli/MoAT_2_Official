
import torch 
from torch import nn
import networkx as nx
import math

import utils

import random
import time

#this is the ipfp model (for a general categorical paramaterization)

from tqdm import tqdm
from cont_copula import *


EPS=1e-7

class MoAT(nn.Module):

    ########################################
    ###     Parameter Initialization     ###
    ########################################

    def __init__(self, n, x, num_classes, device='cpu', K = 1):
        super().__init__()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("on device ", device)
        self.n = x.shape[1]
        if torch.isnan(x).any():
            print("Found NaNs in features (X)!")
            exit(1)

        n = self.n
        self.l = num_classes

        print('initializing params ...')
        with torch.no_grad():
            m = x.shape[0]  # samples

            # estimate marginals from data
            x = x.to(device)

            # pairwise marginals
            print(f"Allocating E with shape: ({n}, {n}, {self.l}, {self.l})")

            E = torch.zeros(n, n, self.l, self.l).to(device)  
            block_size = (2 ** 30) // (n * n * self.l * self.l)
            for block_idx in tqdm(range(0, m, block_size)):
                block_size_ = min(block_size, m - block_idx)
                x_block = x[block_idx:block_idx + block_size_]
                x_2d = torch.zeros(block_size_, n, n, self.l, self.l).to(device)
                x1, x2 = x_block.unsqueeze(2), x_block.unsqueeze(1)
                for l1 in range(self.l):
                    for l2 in range(self.l):
                        x1l1 = (x1 == l1).float()                        
                        x2l2 = (x2 == l2).float()                        
                        x_2d[:, :, :, l1, l2] = torch.matmul(x1l1, x2l2)

                E += torch.sum(x_2d, dim=0)  # shape: [n, n, l, l]
            #E = (E+1) / float(m + 2)
            E = (E+EPS) / float(m+EPS)
            #new init of E_compress
            E_compress = E.clone()
            E_compress = E_compress.unsqueeze(0)
            E_compress = E_compress.expand(K, -1, -1, -1, -1).clone()
            epsilon_noise = 0.000 * torch.randn_like(E_compress) #forget the mixture idea for now
            E_compress = E_compress + epsilon_noise
            E_compress = torch.clamp(E_compress, min=EPS)  # avoid negative probs
            E_compress = E_compress / E_compress.sum(dim=(-2, -1), keepdim=True)
            mix_ws = torch.full((K,), 1.0 / K)
            mix_ws = torch.log(mix_ws)

            # univariate marginals initialization
            V = torch.zeros(n, self.l).to(device)
            for i in range(self.l):
                cnt = torch.sum(x == i, dim=0)  # count how many times i appears in each column
                V[:, i] = (cnt) / (float(m)+EPS)

            V_compress = V.clone()

            E_compress = torch.clamp(E_compress, min=EPS)  # to avoid log(0) or div by 0
            V_compress = torch.clamp(V_compress, min = EPS)

            V_compress = torch.log(V_compress)
            E_compress = torch.log(E_compress)

            print('computing MI ...')
            E_new = torch.maximum(E, torch.ones(1, device=device) * EPS).to(device) #Pairwise joints
            left = V.unsqueeze(1).unsqueeze(-1)  # shape: [n, 1, l, 1]
            right = V.unsqueeze(1).unsqueeze(0)  # shape: [1, n, 1, l]

            # gives tensor n,n,l,l -> pairwise mutual info distributions (assuming independence for baseline comparison)
            V_new = torch.maximum(left * right, torch.ones(1, device=device)* EPS).to(device)
            MI = torch.sum(torch.sum(E_new * torch.log(E_new / V_new), dim=-1), dim=-1)
            MI += EPS

            #ENSURE IN RANGE OF 0-1
            MI_max = (MI.max()+EPS).unsqueeze(0).unsqueeze(0)
            if MI_max >= 1: MI = MI / (MI_max)

            MI = torch.special.logit(MI)

        # W stores the edge weights
        self.W = nn.Parameter(MI, requires_grad=True)
        self.K = K
        self.mix_ws = nn.Parameter(mix_ws, requires_grad = True)
        self.E_compress = nn.Parameter(E_compress, requires_grad = True)
        self.V_compress = nn.Parameter(V_compress, requires_grad=True)
        self.softmax = nn.Softmax(dim=1)  # define softmax layer here

    ########################################
    ###             Inference            ###
    ########################################

    def forward(self, x, epoch = 0, which_moat = 0, step_cnt = 1):
        batch_size, d = x.shape
        n, W, V_compress, E = self.n, self.W, self.softmax(self.V_compress),torch.exp(self.E_compress[which_moat]) #convert back to raw probabilities
        E = E / E.sum(dim=(-2, -1), keepdim=True)
        EPS = torch.tensor(1e-7, device=E.device) 
        
        '''
        if epoch > 25: max_iters = 300
        else: max_iters = 250
        stage_iters = 50
        mae_thresh = 3e-5  # your desired threshold

        active_mask = torch.ones((self.n, self.n), device = E.device, dtype = torch.bool)
        diag_mask = ~torch.eye(n, dtype=torch.bool, device=E.device)
        active_mask = active_mask & diag_mask

        #dynamically adjust num of iterations later
        for stage in range(max_iters // stage_iters):
            i_idx, j_idx = torch.nonzero(active_mask, as_tuple=True)
            A, B = V_compress[i_idx], V_compress[j_idx]

            if i_idx.numel() == 0:
                #print("Stage ", stage, "all distributions have converged — breaking.")
                break 

            for _ in range(stage_iters):
                row_marg = E.sum(dim=3, keepdim=True) + EPS
                E[i_idx, j_idx] = E[i_idx, j_idx] * (A.unsqueeze(-1) / row_marg[i_idx, j_idx])
                col_marg = E.sum(dim=2, keepdim=True) + EPS  # [num_active, 1, l]
                E[i_idx, j_idx] = E[i_idx, j_idx] * (B.unsqueeze(1) / col_marg[i_idx, j_idx])
                E[i_idx, j_idx].div_(E[i_idx, j_idx].sum(dim=(-2, -1), keepdim=True) + EPS)

            row_margs = E.sum(dim=-1)  # [n, n, l]
            col_margs = E.sum(dim=-2)  # [n, n, l]
            V_i = V_compress[:, None, :]  # [n, 1, l]
            V_j = V_compress[None, :, :]  # [1, n, l]

            row_diff = torch.abs(row_margs - V_i)
            col_diff = torch.abs(col_margs - V_j)
            row_diff = row_diff * diag_mask.unsqueeze(-1)
            col_diff = col_diff * diag_mask.unsqueeze(-1)
            mae = (row_diff + col_diff)/2.0
            mae_per_pair = mae.mean(dim=-1)

            still_not_converged = (mae_per_pair > mae_thresh) & diag_mask.bool()
            active_mask = still_not_converged
        
        '''
        A_b = V_compress[:, None, :, None]  # [n, 1, l, 1]
        B_b = V_compress[None, :, None, :]  # [1, n, 1, l]
        for _ in range(2): 
            row_marg = E.sum(dim=3, keepdim=True)   # [n, n, l, 1]
            E.mul_(A_b / row_marg)
            col_marg = E.sum(dim=2, keepdim=True)   # [n, n, 1, l]
            E.mul_(B_b / col_marg)
            E.div_(E.sum(dim=(-2, -1), keepdim=True))

        if step_cnt % 500 == 0:
            # === Marginal Consistency Diagnostics ===
            row_margs = E.sum(dim=-1)  # [n, n, l]
            col_margs = E.sum(dim=-2)  # [n, n, l]
            V_i = V_compress[:, None, :]  # [1, n, 1, l]
            V_j = V_compress[None, :, :]  # [1, 1, n, l]

            row_diff = torch.abs(row_margs - V_i)
            col_diff = torch.abs(col_margs - V_j)
            diag_mask = 1 - torch.eye(self.n, device=E.device).unsqueeze(-1)  # [n, n, 1] ?? 
            row_diff = row_diff * diag_mask
            col_diff = col_diff * diag_mask

            row_mae = row_diff.mean().item()
            col_mae = col_diff.mean().item()
            row_median = row_diff.flatten().median().item()
            col_median = col_diff.flatten().median().item()
            max_row = row_diff.max().item()
            max_col = col_diff.max().item()

            row_difprint = row_diff.mean(dim = -1) #nxn 
            '''
            for i in range(8):
                for j in range(8):
                    if row_difprint[i,j] >= 0.0050:  
                        print("ABOVE THRESH ", i," ", j,": ")
                        print(E[i,j])
                        print("target row ", V_compress[i])
                        print("target col ", V_compress[j])
            '''

            print("\n=== Marginal Consistency Check (in FWD) ===")
            print(f"Mean abs (row, col) marginal diff (E vs V): {row_mae:.6f}, {col_mae:.6f}")
            print(f"Median abs (row, col) marginal diff (E vs V): {row_median:.6f}, {col_median:.6f}")
            print(f"Max (row, col) marginal diff: {max_row:.6f}, {max_col:.6f}")

        V = V_compress.clone()
        E_mask = (1.0 - torch.diag(torch.ones(n)).unsqueeze(-1).unsqueeze(-1)).to(E.device) #broadcasts to 1, 1, n, n diag matrix (zeroes out Xi, Xi)
        E = E * E_mask
        E=torch.clamp(E,0,1)

        W = torch.sigmoid(W)
        W = torch.tril(W, diagonal=-1)
        W = torch.transpose(W, 0, 1) + W

        # det(principal minor of L_0) gives the normalizing factor over the spanning trees
        L_0 = -W + torch.diag_embed(torch.sum(W, dim=1))

        Pr = V[torch.arange(n).unsqueeze(0), x]
        assert not torch.isnan(Pr).any(), "NaN in Pr"

        P = E[torch.arange(n).unsqueeze(0).unsqueeze(-1),
                torch.arange(n).unsqueeze(0).unsqueeze(0),
                x.unsqueeze(-1),
                x.unsqueeze(1)] # E[i, j, x[idx, i], x[idx, j]]
        assert not torch.isnan(P).any(), "NaN in P (from E lookup)"

#        print("matmul")
        torch.cuda.synchronize()
        start = time.time()
        P = P / torch.matmul(Pr.unsqueeze(2), Pr.unsqueeze(1)) # P: bath_size * n * n
        assert not torch.isnan(P).any(), "NaN in normalized P"

        torch.cuda.synchronize()
#        print("matmul time elapsed:", time.time() - start)

        W = W.unsqueeze(0) # W: 1 * n * n; W * P: batch_size * n * n
        L = -W * P + torch.diag_embed(torch.sum(W * P, dim=2))  # L: batch_size * n * n
        log_L, log_L0 = torch.log(L[:, 1:, 1:]), torch.log(L_0[1:, 1:])
        assert not torch.isnan(L).any(), "NaN in L"
         
        #print("log det ")
        torch.cuda.synchronize()
        start = time.time()
        #y = torch.sum(torch.log(Pr), dim=1) + torch.logdet(L[:, 1:, 1:]) - torch.logdet(L_0[1:, 1:])
        try:
            y = torch.sum(torch.log(Pr + 1e-7), dim=1)
            y += torch.logdet(L[:, 1:, 1:] + 1e-7)
            y -= torch.logdet(L_0[1:, 1:] + 1e-7)
        except RuntimeError as e:
            print("logdet failed:", e)
            exit(0)

        torch.cuda.synchronize()
        #print("logdet time elapsed:", time.time() - start)

        if y[y != y].shape[0] != 0:
            print("NaN!")
            exit(0)
        return y

    ########################################
    ### Methods for Sampling Experiments ###
    ########################################

    # can be repurposed to just return samples!

    # spanning tree distribution normalization constant
    def log_Z(self):
        W = torch.sigmoid(self.W)
        W = torch.tril(W, diagonal=-1)
        W = torch.transpose(W, 0, 1) + W
        L_0 = -W + torch.diag_embed(torch.sum(W, dim=1))
        return torch.logdet(L_0[1:, 1:]).item()

    def sample_spanning_tree(self,G,W):
        st=nx.algorithms.random_spanning_tree(G,weight='weight').edges
        parents=utils.get_parents(st,self.n)

        # unnormalized weight of sampled spanning tree
        # log_w=1
        # for (i,j) in st:
        #     log_w+=math.log(W[i][j])
        # normalized weight of spanning tree
        # log_wst=log_w - Z

        return st, parents #, log_wst

    # returns V,E,W after projecting to right space
    def get_processed_parameters(self):
        n, V_compress, E_compress = self.n, torch.sigmoid(self.V_compress),torch.sigmoid(self.E_compress)
        upper_bound = torch.minimum(V_compress.unsqueeze(0), V_compress.unsqueeze(-1))
        lower_bound = torch.maximum(V_compress.unsqueeze(-1) + V_compress.unsqueeze(0) - 1.0,
                        torch.zeros(E_compress.shape).to(V_compress.device)+EPS)

        E_compress = E_compress * ((upper_bound - lower_bound)+EPS) + lower_bound

        V1 = V_compress # n
        V0 = 1 - V_compress
        V = torch.stack((V0, V1), dim=1) # n * 2
        V=V.cpu().detach().numpy()

        E11 = E_compress # n * n
        E01 = V1.unsqueeze(0) - E11
        E10 = V1.unsqueeze(-1) - E11
        E00 = 1 - E01 - E10 - E11
        E0 = torch.stack((E00, E01), dim=-1) # n * n * 2
        E1 = torch.stack((E10, E11), dim=-1) # n * n * 2
        E = torch.stack((E0, E1), dim=2)
        E_mask = (1.0 - torch.diag(torch.ones(n)).unsqueeze(-1).unsqueeze(-1)).to(E.device)
        E = E * E_mask
        E=E.cpu().detach().numpy()

        W = torch.sigmoid(self.W)
        W = torch.tril(W, diagonal=-1)
        W = torch.transpose(W, 0, 1) + W

        return V,E,W


    def get_true_marginals(self,evidence):
        n=self.n
        true_marginals=[1 for i in range(n)]
        for i in range(n):
            if evidence[i]!=-1:
                true_marginals[i]=evidence[i]
                continue
            evidence[i]=0
            data=utils.generate_marginal_terms(n,evidence)
            p_0=torch.sum(torch.exp(self.forward(data))).item()
            evidence[i]=1
            data=utils.generate_marginal_terms(n,evidence)
            p_1=torch.sum(torch.exp(self.forward(data))).item()

            true_marginals[i]=p_1/(p_0+p_1)
            evidence[i]=-1
        return true_marginals

    def get_importance_samples(self,evidence,num_samples=1):
        num_missing=evidence.count(-1)
        n=self.n
        V,E,W=self.get_processed_parameters()

        true_marginals=self.get_true_marginals(evidence)

        marginals=[0 for i in range(n)]
        norm=0

        klds=[]
        wts=[]
        G = nx.Graph()
        for i in range(self.n):
            for j in range(i+1,self.n):
                G.add_edge(i,j,weight=W[i][j].item())

        data=utils.generate_marginal_terms(n,evidence)
        m_e=torch.sum(torch.exp(self.forward(data))).item()

        for it in range(num_samples):
            st, parents = self.sample_spanning_tree(G,W)
            res,wt=utils.sample_from_tree_factor_autoregressive(n,parents,E,V,evidence)
            for i in range(n):
                if res[i]:
                    marginals[i]+=wt
            norm+=wt
            wts.append(wt/m_e)
            approximate_marginals=[marginals[i]/norm for i in range(n)]
            kld=utils.kld(true_marginals,approximate_marginals)/num_missing
            klds.append(kld)

        return klds,wts

    def get_collapsed_importance_samples(self,evidence,num_samples=1):
        num_missing=evidence.count(-1)
        n=self.n
        V,E,W=self.get_processed_parameters()

        true_marginals=self.get_true_marginals(evidence)

        marginals=[0 for i in range(n)]
        norm=0

        klds=[]
        wts=[]
        G = nx.Graph()
        for i in range(self.n):
            for j in range(i+1,self.n):
                G.add_edge(i,j,weight=W[i][j].item())

        data=utils.generate_marginal_terms(n,evidence)
        m_e=torch.sum(torch.exp(self.forward(data))).item()

        for it in range(num_samples):
            st, parents = self.sample_spanning_tree(G,W)
            wt,marginal_vector=utils.sample_from_tree_parallel(n,parents,E,V,evidence)
            for i in range(n):
                marginals[i]+=wt*max(0,min(1,marginal_vector[i]))
            norm+=wt
            wts.append(wt/m_e)
            approximate_marginals=[marginals[i]/norm for i in range(n)]
            kld=utils.kld(true_marginals,approximate_marginals)/num_missing
            klds.append(kld)


        return klds,wts

    def get_gibbs_samples(self,evidence,burn_in=10,num_samples=1):
        num_missing=evidence.count(-1)
        n=self.n
        V,E,W=self.get_processed_parameters()

        true_marginals=self.get_true_marginals(evidence)

        marginals=[0 for i in range(n)]
        norm=0

        cur=evidence.copy()
        for i in range(n):
            if cur[i]==-1:
                cur[i]=random.randint(0, 1)

        idx=-1
        klds=[]
        for it in range(burn_in+num_samples):
            evi=cur.copy()
            for idx in range(n):
                if evidence[idx]!=-1:
                    continue
                evi[idx]=-1
                # note that this generates the 0 term first followed by the 1 term
                data=utils.generate_marginal_terms(n,evi)
                p=torch.exp(self.forward(data))
                cur[idx]=0 if random.uniform(0, 1)<=p[0].item()/(p[0].item()+p[1].item()) else 1
                evi[idx]=cur[idx]

            if it<burn_in:
                continue

            for i in range(n):
                if cur[i]:
                    marginals[i]+=1
            norm+=1

            if it >= burn_in:
                approximate_marginals=[marginals[i]/norm for i in range(n)]
                kld=utils.kld(true_marginals,approximate_marginals)/num_missing
                klds.append(kld)

        return klds

