import torch
import math

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

#IPFP algo (for comparison)
def ipfp(M, a, b, its = 20):
  for _ in range(its):
    row_margs = M.sum(dim = 1, keepdim= True)
    M = M * (a.unsqueeze(1) / row_margs)
    col_margs = M.sum(dim = 0, keepdim= True)
    M = M * (b.unsqueeze(0) / col_margs)
  P = M
  return P


#PRINTOUTS
'''
print("Current row marginals: ", E_rows)
print("Current col marginals: ", E_cols)
print("Target row marginals: ", R)
print("Target col marginals: ", C)
print("E_rows cumsum: ", E_rows_cumsum)
print("E_cols cumsum: ", E_cols_cumsum)


'''
def G1(x1, R, R_cumsum):
    l = R.shape[0]
    x1 = x1.to(R.device).float()
    R = R.to(dtype=torch.float32)
    R_cumsum = R_cumsum.to(dtype=torch.float32)

    if x1.dim() == 1:
        x1 = x1.unsqueeze(-1)

    vals = torch.zeros((x1.shape[0], 1), dtype=torch.float32, device=x1.device)
    ml = (x1 == l)
    vals[ml] = 1.0

    fl = torch.floor(x1)
    fl_long = fl.long()

    not_l_mask = (~ml).squeeze(-1) & (fl_long.squeeze(-1) != 0)
    st = torch.zeros((x1.shape[0], 1), dtype=torch.float32, device=x1.device)

    idx = fl_long[not_l_mask] - 1
    st[not_l_mask] = R_cumsum[idx]
    vals[~ml] = st[~ml] + R[fl_long[~ml].squeeze(-1)] * (x1[~ml] - fl[~ml])

    return vals


'''
def G2(x2, C, C_cumsum):
 x2 = x2.to(C.device).float()
 if x2.dim() == 1: x2 = x2.unsqueeze(-1)
 vals = torch.zeros((x2.shape[0], 1), dtype=torch.float32, device=x2.device)
 ml = (x2 == l)
 vals[ml] = 1.0
 fl = torch.floor(x2)
 fl_long = fl.long()
 C_cumsum = C_cumsum.to(dtype=torch.float32)

 not_l_mask = (~ml) & (fl_long != 0)
 st = torch.zeros_like(x2, dtype=torch.float32, device=x2.device)
 st[not_l_mask] = C_cumsum[fl_long[not_l_mask] - 1]
 vals[~ml] = st[~ml] + C[fl_long[~ml]] * (x2[~ml] - fl[~ml])
 return vals
'''

def G2(x2, C, C_cumsum):
    l = C.shape[0]
    x2 = x2.to(C.device).float()
    C = C.to(dtype=torch.float32)
    C_cumsum = C_cumsum.to(dtype=torch.float32)

    if x2.dim() == 1:
        x2 = x2.unsqueeze(-1)

    vals = torch.zeros((x2.shape[0], 1), dtype=torch.float32, device=x2.device)
    ml = (x2 == l)
    vals[ml] = 1.0

    fl = torch.floor(x2)
    fl_long = fl.long()

    not_l_mask = (~ml).squeeze(-1) & (fl_long.squeeze(-1) != 0)
    st = torch.zeros((x2.shape[0], 1), dtype=torch.float32, device=x2.device)

    idx = fl_long[not_l_mask] - 1
    st[not_l_mask] = C_cumsum[idx]

    vals[~ml] = st[~ml] + C[fl_long[~ml].squeeze(-1)] * (x2[~ml] - fl[~ml])
    return vals

def inv_cdf_batch(u, cumsum, pmf):
    device = cumsum.device
    if u.dim() == 1: u = u.unsqueeze(-1).to(device)  # shape (N, 1)
    idx = torch.searchsorted(cumsum, u, right=True)
    idx = idx.clamp(max=len(pmf)-1)

    below = torch.where(idx > 0, cumsum[idx - 1], torch.zeros_like(u))
    delta = torch.clamp(pmf[idx], min = 1e-7)
    offset = (u - below) / delta
    return idx.squeeze(-1) + offset.squeeze(-1)

#Note: KLD cannot exceed 2logk (or log k^2) where k is the number of categories
#scale against that


def F_batch(x1, x2, E, E_cumsum):
    device = E.device
    # E is (l, l)
    if x2.dim() == 1: x2 = x2.unsqueeze(-1).to(device)
    if x1.dim() == 1: x1 = x1.unsqueeze(-1).to(device)
    m = x1.floor().long()
    n = x2.floor().long()

    m = torch.clamp(m, max=E.size(0)-1)
    n = torch.clamp(n, max=E.size(1)-1)
    dx = x1 - m.float()
    dy = x2 - n.float()

    '''
    def safe_gather(E_cumsum, m_idx, n_idx):
        mask = (m_idx >= 0) & (n_idx >= 0)
        vals = torch.zeros((m.shape[0], 1), device=device)
        if mask.any():
            valid_indices = torch.stack([m_idx[mask], n_idx[mask]], dim=0)  # shape (2, N)
            vals[mask] = E_cumsum[valid_indices[0], valid_indices[1]]
        return vals
    '''
    def safe_gather(E_cumsum, m_idx, n_idx):
        mask = (m_idx >= 0) & (n_idx >= 0)
        vals = torch.zeros((m_idx.shape[0], 1), dtype=E_cumsum.dtype, device=E_cumsum.device)

        if mask.any():
            valid_indices = torch.stack([m_idx[mask], n_idx[mask]], dim=0)  # shape (2, N)
            gathered = E_cumsum[valid_indices[0], valid_indices[1]]
            vals[mask] = gathered.to(dtype=vals.dtype)

        return vals

    A = safe_gather(E_cumsum, m-1, n-1)
    B = safe_gather(E_cumsum, m-1, n)
    C_val = safe_gather(E_cumsum, m, n-1)
    return A + (C_val-A)*(dx) + (B-A)*(dy) + E[m, n]*(dx)*(dy)

def Cop_batch(u1, u2, E_rows, E_rows_cumsum, E_cols, E_cols_cumsum, E, E_cumsum):
    return F_batch(inv_cdf_batch(u1, E_rows_cumsum, E_rows), inv_cdf_batch(u2, E_cols_cumsum, E_cols), E, E_cumsum)  # or pass E if not global

def G(x1, x2, E_rows, E_rows_cumsum, E_cols, E_cols_cumsum, R, R_cumsum, C, C_cumsum, E, E_cumsum):
  return Cop_batch(G1(x1, R, R_cumsum), G2(x2, C, C_cumsum), E_rows, E_rows_cumsum, E_cols, E_cols_cumsum, E, E_cumsum)

def G_batch(E, E_rows, E_rows_cumsum, E_cols, E_cols_cumsum, R, R_cumsum, C, C_cumsum, E_cumsum):
  #E_cop = [G(i + 1, j + 1) - (G(i, j + 1) + G(i + 1, j) - G(i, j)) for i in range(l) for j in range(l)]
  l = E.shape[0]
  I, J = torch.meshgrid(torch.arange(l), torch.arange(l), indexing='ij')
  I = I.flatten()  # shape (l^2,)
  J = J.flatten()
  I0, I1 = I, I+1
  J0, J1 = J, J+1

  G_11 = G(I1, J1, E_rows, E_rows_cumsum, E_cols, E_cols_cumsum, R, R_cumsum, C, C_cumsum, E, E_cumsum)  # G(i+1, j+1)
  G_01 = G(I0, J1, E_rows, E_rows_cumsum, E_cols, E_cols_cumsum, R, R_cumsum, C, C_cumsum, E, E_cumsum)  # G(i, j+1)
  G_10 = G(I1, J0, E_rows, E_rows_cumsum, E_cols, E_cols_cumsum, R, R_cumsum, C, C_cumsum, E, E_cumsum)  # G(i+1, j)
  G_00 = G(I0, J0, E_rows, E_rows_cumsum, E_cols, E_cols_cumsum, R, R_cumsum, C, C_cumsum, E, E_cumsum)  # G(i, j)
  E_cop_flat = G_11 - G_01 - G_10 + G_00
  E_cop = E_cop_flat.view(l, l)
  return E_cop

def KLD(P1, P2):
    P1 = torch.clamp(P1, min=1e-7)
    P2 = torch.clamp(P2, min=1e-7)
    return (P1 * (P1 / P2).log()).sum()

'''
print("KLD (E_original, E_copula) ", KLD(E, E_cop))
print("KLD (E_original, E_ipfp)", KLD(E, E_ipfp))
print("KLD (E_ipfp, E_copula)", KLD(E_ipfp, E_cop), KLD(E_cop, E_ipfp))
'''
