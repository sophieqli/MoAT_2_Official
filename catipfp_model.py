
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
from catipfp_utils import *


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

            V_new = torch.maximum(left * right, torch.ones(1, device=device)* EPS).to(device)
            MI = torch.sum(torch.sum(E_new * torch.log(E_new / V_new), dim=-1), dim=-1)
            MI += EPS

            #ENSURE IN RANGE OF 0-1
            self.MI_unorm = MI
            MI_max = (MI.max()).unsqueeze(0).unsqueeze(0)
            if torch.isnan(MI).any():
                print("MI contains NaNs! BEFORE RESCALIGN ")
            if MI_max >= 1: 
                print("MI MAX ", MI_max, " RESCALED ")
                MI = MI / (MI_max)
            
            MI = torch.clamp(MI, EPS, 1-EPS)
            MI = torch.special.logit(MI )

        # W stores the edge weights
        self.b4 = 0
        self.W = nn.Parameter(MI, requires_grad=True)
        self.K = K
        self.mix_ws = nn.Parameter(mix_ws, requires_grad = True)
        self.E_compress = nn.Parameter(E_compress, requires_grad = True)
        self.V_compress = nn.Parameter(V_compress, requires_grad=True)
        self.softmax = nn.Softmax(dim=1)  
        print("shapes of params: ")
        print(self.W.shape, self.E_compress.shape, self.V_compress.shape)

    ########################################
    ###             Inference            ###
    ########################################

    def forward(self, x, epoch = 0, which_moat = 0, step_cnt = 1):
        batch_size, d = x.shape
        n, W, V_compress, E = self.n, self.W, self.softmax(self.V_compress),torch.exp(self.E_compress[which_moat]) #convert back to raw probabilities
        E = E / E.sum(dim=(-2, -1), keepdim=True)
        EPS = torch.tensor(1e-7, device=E.device) 
        
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
            marg_consistency_diagnostics(V_compress, E, self.n)

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

        P = P / torch.matmul(Pr.unsqueeze(2), Pr.unsqueeze(1)) # P: bath_size * n * n
        assert not torch.isnan(P).any(), "NaN in normalized P"

        torch.cuda.synchronize()

        W = W.unsqueeze(0) # W: 1 * n * n; W * P: batch_size * n * n
        L = -W * P + torch.diag_embed(torch.sum(W * P, dim=2))  # L: batch_size * n * n
        log_L, log_L0 = torch.log(L[:, 1:, 1:]), torch.log(L_0[1:, 1:])
        assert not torch.isnan(L).any(), "NaN in L"
         
        #y = torch.sum(torch.log(Pr), dim=1) + torch.logdet(L[:, 1:, 1:]) - torch.logdet(L_0[1:, 1:])
        try:
            y = torch.sum(torch.log(Pr + 1e-7), dim=1)
            y += torch.logdet(L[:, 1:, 1:] + 1e-7)
            y -= torch.logdet(L_0[1:, 1:] + 1e-7)
        except RuntimeError as e:
            print("logdet failed:", e)
            exit(0)

        if y[y != y].shape[0] != 0:
            print("NaN!")
            exit(0)
        return y


    def ipfp_yxk(E_new, E_compress_old, i, j, V_y, num_iter=150):
        """
        Batched IPFP update for all (Y, X_k) pairs where Y is last index in E_new.
        Enforces:
          - P(Y) = V_y
          - P(X_i, X_k) from old E_compress
          - P(X_j, X_k) from old E_compress
        """
        K, n_new, _, l, _ = E_new.shape
        y_idx = n_new - 1
        num_pairs = n_new - 1  # variables other than Y

        # 1) Extract (Y, X_k) blocks: shape (K, num_pairs, 4, l)
        yxk = E_new[:, y_idx, :y_idx, :, :].clone()

        # 2) Build targets for p(X_i, X_k) and p(X_j, X_k) from old E_compress
        keep_idx = [idx for idx in range(E_compress_old.shape[1]) if idx not in (i, j)]
        # targets shape (K, num_pairs, 2, l)
        targets_xi_xk = E_compress_old[:, i, keep_idx, :2, :].clone()
        targets_xj_xk = E_compress_old[:, j, keep_idx, :2, :].clone()

        # Ensure float and normalized-ish
        yxk = yxk.clamp(min=eps)
        targets_xi_xk = targets_xi_xk.clamp(min=eps)
        targets_xj_xk = targets_xj_xk.clamp(min=eps)
        V_y = V_y.to(yxk.device).view(1, 1, 4)  # (1,1,4) for broadcasting

        for _ in range(num_iter):
            # --- Constraint 1: match P(Y) ---
            cur_pY = yxk.sum(dim=3)  # (K, n_new-1, 4)
            scale_Y = V_y.view(1, 1, 4) / (cur_pY + 1e-12)
            yxk *= scale_Y.unsqueeze(-1)

            # --- Constraint 2: match P(X_i, X_k) ---
            cur_xi0 = yxk[:,:,[0,1],:].sum(dim = 2) # (K, num_pairs, l)
            cur_xi0 = yxk[:,:,[2,3],:].sum(dim = 2) # (K, num_pairs, l)
            cur_pXiXk = torch.stack([cur_xi0, cur_xi1], dim=2)  # (K, num_pairs, 2, l)
            scale_xi = targets_xi_xk/cur_pXiXk 
            yxk[:, :, [0, 1], :] = yxk[:, :, [0, 1], :] * scale_xi[:, :, 0, :].unsqueeze(2)
            yxk[:, :, [2, 3], :] = yxk[:, :, [2, 3], :] * scale_xi[:, :, 1, :].unsqueeze(2)

            # --- Constraint 3: match P(X_j, X_k) ---
            cur_xj0 = yxk[:, :, [0, 2], :].sum(dim=2)  # (K, num_pairs, l)
            cur_xj1 = yxk[:, :, [1, 3], :].sum(dim=2)  # (K, num_pairs, l)
            cur_pXjXk = torch.stack([cur_xj0, cur_xj1], dim=2)  # (K, num_pairs, 2, l)
            scale_xj = targets_xj_xk / (cur_pXjXk + eps)  # (K, num_pairs, 2, l)
            yxk[:, :, [0, 2], :] = yxk[:, :, [0, 2], :] * scale_xj[:, :, 0, :].unsqueeze(2)
            yxk[:, :, [1, 3], :] = yxk[:, :, [1, 3], :] * scale_xj[:, :, 1, :].unsqueeze(2)

            #Re-normalize (optional)
            yxk = yxk / (yxk.sum(dim=(2,3), keepdim=True))

        # write back into E_new symmetrically
        E_new[:, y_idx, :y_idx, :, :] = yxk
        E_new[:, :y_idx, y_idx, :, :] = yxk.transpose(2, 3)

        return E_new

    def contract_edge_b2tob4_params(self, i, j):
        """
        Contract two binary vars X_i, X_j into a base-4 var Y = 2*X_i + X_j.
        Assumes V_compress: (n, l), E_compress: (n, n, l, l) with l=4 initially.
        Keeps Y as the last variable.
        """
        EPS = 1e-7
        assert self.V_compress.shape[1] == 4, "Expected l=4 categories"

        K, n, _, l, _ = self.E_compress.shape
        #transform self.E_compress into raw values (from log space)
        self.E_compress = torch.exp(self.E_compress)
        self.E_compress = self.E_compress / self.E_compress.sum(dim=(-2, -1), keepdim=True)
        E_compress_old = self.E_compress

        # 1. Create univariate marginals for Y directly from joint(X_i, X_j)
        joint_ij = self.E_compress[-1, i, j, :2, :2]  # shape (2, 2) since binary
        V_y = joint_ij.reshape(-1)  # flatten into length-4 vector

        # 2. Create pairwise marginals Y vs every other X_k
        E_yk_allmix = []
        for mix in range(K):
            E_yk_list = []
            for k in range(n):
                if k in (i, j):
                    continue

                # p(Y, X_k) ∝ p(X_i, X_k) and p(X_j, X_k)
                p_ik = self.E_compress[mix,i, k, :2, :]  # shape (2, l)
                p_jk = self.E_compress[mix,j, k, :2, :]  # shape (2, l)
                p_k  = self.V_compress[k, :]          # (l,)

                E_yk = torch.zeros((4, l), device=self.V_compress.device)
                for xi in range(2):
                    for xj in range(2):
                        y_val = 2*xi + xj
                        E_yk[y_val, :] = (
                            p_ik[xi, :] * p_jk[xj, :] / (p_k + 1e-12)  # avoid div by 0
                        ) * self.E_compress[mix, i,j,xi,xj]
                # Renormalize to match univariate V_y[y_val] on each row
                E_yk_list.append(E_yk)
            E_yk_allmix.append(E_yk_list)

        # 3. Build new V_compress with Y at end
        keep_idx = [k for k in range(n) if k not in (i, j)]
        V_new = self.V_compress[keep_idx]
        V_new = torch.cat([V_new, V_y.unsqueeze(0)], dim=0)

        # 4. Build new E_compress
        E_new_allmix = []
        for mix in range(K):
            # Keep existing unchanged pairs
            E_new_mix = self.E_compress[mix][keep_idx][:, keep_idx, :, :]  # (n-2, n-2, l, l)

            # Append Y row (pairs Y,X_k)
            E_y_row = torch.stack(E_yk_allmix[mix], dim=0)  # (n-2, 4, l)
            E_new_mix = torch.cat([E_new_mix, E_y_row.unsqueeze(1)], dim=1)  # col for Y

            # Append Y column (pairs X_k,Y) and self
            E_y_col = torch.stack(E_yk_allmix[mix], dim=0).transpose(1, 2)  # (n-2, l, 4)
            E_y_self = torch.zeros((4, 4), device=self.V_compress.device)
            torch.diagonal(E_y_self)[:] = V_y

            E_new_mix = torch.cat(
                [E_new_mix,
                 torch.cat([E_y_col.unsqueeze(0), E_y_self.unsqueeze(0)], dim=0)],
                dim=0
            )

            E_new_allmix.append(E_new_mix)

        E_new = torch.stack(E_new_allmix, dim=0)  # (K, n-1, n-1, l, l)
        E_new = torch.clamp(E_new, min=EPS)  # to avoid log(0) or div by 0
        #IPFP MATCHING for joint 
        E_new_ipfp = ipfp_yxk(E_new, E_compress_old, i, j, V_y, num_iter=150)
        self.E_compress = nn.Parameter(torch.log(E_new_ipfp), requires_grad = True)
        V_new = torch.clamp(V_new, min=EPS)  # to avoid log(0) or div by 0
        self.V_compress = nn.Parameter(torch.log(V_new), requires_grad = True)

        # Update weights W — keep others, sum for new Y
        W_new = self.W[keep_idx][:, keep_idx]
        Wy = torch.tensor(
            [self.W[k, i] + self.W[k, j] for k in range(n) if k not in (i, j)],
            device=self.W.device
        )
        # Add Y as last row/col
        W_new = torch.cat([W_new, Wy.unsqueeze(1)], dim=1)
        W_new = torch.cat([W_new, torch.cat([Wy, torch.tensor([0.0], device=self.W.device)]).unsqueeze(0)], dim=0)

        W_max = (W_new.max()).unsqueeze(0).unsqueeze(0)
        if W_max >= 1: 
            W_new = W_new / (W_max)
            W_new = torch.clamp(W_new, EPS, 1-EPS)
            W_new = torch.special.logit(W_new)
            
        self.W = nn.Parameter(W_new, requires_grad = True)

        self.n -= 1
        self.b4 += 1

    ########################################
    ### Methods for Sampling Experiments ###
    ########################################

    # spanning tree distribution normalization constant
    def log_Z(self):
        W = torch.sigmoid(self.W)
        W = torch.tril(W, diagonal=-1)
        W = torch.transpose(W, 0, 1) + W
        L_0 = -W + torch.diag_embed(torch.sum(W, dim=1))
        #return torch.logdet(L_0[1:, 1:]).item()
        return torch.logdet(L_0[1:, 1:])

    #NEW: Edge posterior
    def edge_post(self, i, j):
        self.W.requires_grad_(True)
        logZ = self.log_Z()  # should return a scalar tensor now, not .item()
        grad_W = torch.autograd.grad(logZ, self.W, retain_graph=True)[0]
        w_ij = torch.sigmoid(self.W[i, j])
        p_ij = w_ij * grad_W[i, j]
        return p_ij.item()

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

