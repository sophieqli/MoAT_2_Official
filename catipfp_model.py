
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

    def __init__(self, n, x, num_classes, device='cuda', b4 = 0, K = 1):
        super().__init__()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("on device ", device)
        self.device = device
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
            E = (E+EPS) / float(m+EPS)
            #new init of E_compress
            E_compress = E.clone()
            E_compress = E_compress.unsqueeze(0)
            E_compress = E_compress.expand(K, -1, -1, -1, -1).clone()
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
        self.b4 = b4
        self.W = nn.Parameter(MI, requires_grad=True)
        self.K = K
        self.mix_ws = nn.Parameter(mix_ws, requires_grad = True)
        self.E_compress = nn.Parameter(E_compress, requires_grad = True)
        self.V_compress = nn.Parameter(V_compress, requires_grad=True)
        if torch.isnan(self.V_compress).any():
            print("Warning: NaNs detected in V_compress!")

        self.softmax = nn.Softmax(dim=1)  

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
        assert x.min() >= 0 and x.max() < V.shape[1], "x out of bounds for V"

        if torch.isnan(V).any():
            print("Warning: NaNs detected in V_compress! in fwd ")
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

    def freeze_except_Y(self):
        if self.W.grad is None:
            return
        # Create a mask that is 1 for last row and last col, else 0
        mask = torch.zeros_like(self.W.grad)
        mask[-1, :] = 1
        mask[:, -1] = 1
        # Keep gradients only for last row/col
        self.W.grad *= mask

        if self.E_compress.grad is None:
            return
        mask = torch.zeros_like(self.E_compress.grad)
        mask[:, -1, :, :, :] = 1  # last row
        mask[:, :, -1, :, :] = 1  # last col
        self.E_compress.grad *= mask

        if self.V_compress.grad is None:
            return
        mask = torch.zeros_like(self.V_compress.grad)
        mask[-1, :] = 1
        self.V_compress.grad *= mask

    def undo_freeze(self):
        self.W.requires_grad = True
        self.E_compress.requires_grad = True
        self.V_compress.requires_grad = True


    def contract_edge_b2tob4_params(self, i, j, trainx_ij):
        """
        Contract two binary vars X_i, X_j into a base-4 var Y = 2*X_i + X_j.
        Assumes V_compress: (n, l), E_compress: (n, n, l, l) with l=4 initially.
        Keeps Y as the last variable.
        """
        EPS = 1e-7
        assert self.V_compress.shape[1] == 4, "Expected l=4 categories"

        K, n, _, l, _ = self.E_compress.shape
        #transform self.E_compress into raw values (from log space)
        with torch.no_grad():
            new_E = torch.exp(self.E_compress)  # from log space
            new_E = new_E / new_E.sum(dim=(-2, -1), keepdim=True)  # normalize
            self.E_compress.copy_(new_E)  # inplace update of param data
            new_V = self.softmax(self.V_compress)
            self.V_compress.copy_(new_V)
            new_W = torch.sigmoid(self.W)
            self.W.copy_(new_W)
        E_compress_old = self.E_compress.detach().clone()

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
                        '''
                        E_yk[y_val, :] = (
                            p_ik[xi, :] * p_jk[xj, :] / (p_k + 1e-12)  # avoid div by 0
                        ) * self.E_compress[mix, i,j,xi,xj]
                        '''
                        E_yk[y_val, :] = (
                            p_ik[xi, :] * p_jk[xj, :] / (p_k + 1e-12)  # avoid div by 0
                        )
                # Renormalize
                E_yk /= E_yk.sum() + 1e-12

                #TODO: a mix of empirical and prod heuristic!!
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

            # Your combined tensor currently is (n-1, l, l)
            combined = torch.cat([E_y_col, E_y_self.unsqueeze(0)], dim=0)  # shape: (n-1, l, l)
            combined = combined.unsqueeze(1)  # (n-1, 1, l, l)
            combined = combined.transpose(0, 1)  # (1, n-1, l, l)
            # Now concat with E_new_mix along dim=1 (columns)
            E_new_mix = torch.cat([E_new_mix, combined], dim=0)  # final shape: (n-1, n-1, l, l)
            E_new_allmix.append(E_new_mix)

        E_new = torch.stack(E_new_allmix, dim=0)  # (K, n-1, n-1, l, l)
        E_new = torch.clamp(E_new, min=EPS)  # to avoid log(0) or div by 0
        E_new_ipfp = ipfp_yxk(E_new, E_compress_old, i, j, V_y)

        #get empirical distribution -> WITH Xi, Xj DELETED
        assert trainx_ij.shape[1] == n - 1, "dataset formed wrong"
        E_emp = torch.zeros(trainx_ij.shape[1], trainx_ij.shape[1], self.l, self.l).to(self.device)
        m = trainx_ij.shape[0]

        block_size = (2 ** 30) // (n * n * self.l * self.l)
        for block_idx in tqdm(range(0, m, block_size)):
            block_size_ = min(block_size, m - block_idx)
            x_block = trainx_ij[block_idx:block_idx + block_size_]
            x_2d = torch.zeros(block_size_, n-1, n-1, self.l, self.l).to(device)
            x1, x2 = x_block.unsqueeze(2), x_block.unsqueeze(1)
            for l1 in range(self.l):
                for l2 in range(self.l):
                    x1l1 = (x1 == l1).float()                        
                    x2l2 = (x2 == l2).float()                        
                    x_2d[:, :, :, l1, l2] = torch.matmul(x1l1, x2l2)

            E_emp += torch.sum(x_2d, dim=0)  # shape: [n, n, l, l]
        E_emp = (E_emp+EPS) / float(m+EPS)
        E_emp = E_emp.unsqueeze(0) #1,n+1, n+1, l, l

        #Average product heuristic IPFP with empirical 
        E_new_ipfp[:,-1, :] = 0.0*E_new_ipfp[:,-1, : ] + 1.0*E_emp[:,-1, :]
        E_new_ipfp[:, :-1, -1] = 0.0 * E_new_ipfp[:, :-1, -1] + 1.0 * E_emp[:, :-1, -1]

        #WE NEED TO RUN IPFP HERE cuz emp doesn't match params 
        #joint: E_new_ipfp to match V_new
            
        ######
        with torch.no_grad():
            for which_moat in range(self.K):
                A_b, B_b = V_new[:, None, :, None], V_new[None, :, None, :]  
                for _ in range(200):
                    row_marg = E_new_ipfp[which_moat].sum(dim=3, keepdim=True)
                    E_new_ipfp[which_moat].mul_(A_b / row_marg)
                    col_marg = E_new_ipfp[which_moat].sum(dim=2, keepdim=True)
                    E_new_ipfp[which_moat].mul_(B_b / col_marg)
                    E_new_ipfp[which_moat] /= E_new_ipfp[which_moat].sum(dim=(-2, -1), keepdim=True) + EPS
        ########

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
        '''
        E_new = torch.maximum(E_new_ipfp, torch.ones(1, device=self.device) * EPS).to(self.device) #Pairwise joints
        left = V_new.unsqueeze(1).unsqueeze(-1)  # shape: [n, 1, l, 1]
        right = V_new.unsqueeze(1).unsqueeze(0)  # shape: [1, n, 1, l]

        V_new = torch.maximum(left * right, torch.ones(1, device=device)* EPS).to(device)
        W_new = torch.sum(torch.sum(E_new * torch.log(E_new.squeeze(0) / V_new), dim=-1), dim=-1) + EPS
        '''
        W_max = (W_new.max()).unsqueeze(0).unsqueeze(0)
        if W_max >= 1: 
            W_new = W_new / (W_max)
            W_new = torch.clamp(W_new, EPS, 1-EPS)
        W_new = torch.special.logit(W_new).squeeze(0)
            
        self.W = nn.Parameter(W_new, requires_grad = True)

        self.n -= 1
        self.b4 += 1

    ########################################
    ### Methods for Sampling Experiments ###
    ########################################

    # spanning tree distribution normalization constant
    def edge_posts(self):
        #W * grad of log Z w.r.t W, where W is actual weight (sigmoid)
        W_sig = torch.sigmoid(self.W)
        #W_sig.requires_grad_(True)
        W = torch.tril(W_sig, diagonal=-1)
        W = torch.transpose(W, 0, 1) + W
        L_0 = -W + torch.diag_embed(torch.sum(W, dim=1))
        logZ = torch.logdet(L_0[1:, 1:])
        grad_W = torch.autograd.grad(logZ, W_sig)[0]   # gradient w.r.t. actual weights W

        return W_sig * grad_W  # shape (n, n), gradients w.r.t. actual edge weights

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

