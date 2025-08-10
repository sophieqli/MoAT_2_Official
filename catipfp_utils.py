
#funcs 

import torch 
import numpy as np
eps = 1e-7

def ipfp_yxk(E_new, E_compress_old, i, j, V_y, num_iter=200):
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

    yxk_reshaped = yxk.reshape(K, num_pairs, 2, 2, l)  # reshape once
    V_y_expanded = V_y.expand(K, num_pairs, 4)
    # print("V_y_expanded.shape:", V_y_expanded.shape)

    targets_xi_xk_expanded = targets_xi_xk.expand(K, num_pairs, 2, l)
    # print("targets_xi_xk_expanded.shape:", targets_xi_xk_expanded.shape)

    targets_xj_xk_expanded = targets_xj_xk.expand(K, num_pairs, 2, l)
    # print("targets_xj_xk_expanded.shape:", targets_xj_xk_expanded.shape)
    examples = [(0,0), (2,1), (4,0)]

    #print("V_y.shape =", V_y.shape)
    #print("targets_xi_xk.shape =", targets_xi_xk.shape)
    #print("targets_xj_xk.shape =", targets_xj_xk.shape)

    for _ in range(num_iter):
        # Apply Constraint 2: match P(X_i, X_k)
        p_Xi_Xk = yxk_reshaped.sum(dim=3)  # sum over Xj
        scale_xi = targets_xi_xk / (p_Xi_Xk + eps)
        yxk_reshaped *= scale_xi.unsqueeze(3)  # broadcast over Xj dim

        # Apply Constraint 3: match P(X_j, X_k)
        p_Xj_Xk = yxk_reshaped.sum(dim=2)  # sum over Xi
        scale_xj = targets_xj_xk / (p_Xj_Xk + eps)
        yxk_reshaped *= scale_xj.unsqueeze(2)  # broadcast over Xi dim

        # Apply Constraint 1: match P(Y)
        p_Y = yxk_reshaped.sum(dim=4)  # sum over Xk
        p_Y_flat = p_Y.reshape(K, num_pairs, 4)
        scale_Y = V_y.view(1, 1, 4) / (p_Y_flat + eps)
        yxk_reshaped *= scale_Y.reshape(K, num_pairs, 2, 2, 1)  # broadcast over Xk dim

        # Print diagnostics every 50 iterations
        '''
        if (_ + 1) % 50 == 0 or _ == num_iter - 1:
            print(f"\n=== After iteration {_ + 1} ===")
            for dist_idx, pair_idx in examples:
                print(f"\nDistribution {dist_idx}, Pair {pair_idx}")

                print("Target P(Y):", [round(x, 4) for x in V_y_expanded[dist_idx, pair_idx].tolist()])
                print("Target P(X_i, X_k):")
                print([[round(x, 4) for x in row] for row in targets_xi_xk_expanded[dist_idx, pair_idx].tolist()])
                print("Target P(X_j, X_k):")
                print([[round(x, 4) for x in row] for row in targets_xj_xk_expanded[dist_idx, pair_idx].tolist()])

                print("Current P(Y):", [round(x, 4) for x in p_Y_flat[dist_idx, pair_idx].tolist()])
                print("Current P(X_i, X_k):")
                print([[round(x, 4) for x in row] for row in p_Xi_Xk[dist_idx, pair_idx].tolist()])
                print("Current P(X_j, X_k):")
                print([[round(x, 4) for x in row] for row in p_Xj_Xk[dist_idx, pair_idx].tolist()])
        '''

    yxk = yxk_reshaped.reshape(K, num_pairs, 4, l)

    # write back into E_new symmetrically
    E_new[:, y_idx, :y_idx, :, :] = yxk
    E_new[:, :y_idx, y_idx, :, :] = yxk.transpose(2, 3)
    return E_new

def dyn_ipfpits(epoch, V_compress, E, n):
    if epoch > 25: max_iters = 300
    else: max_iters = 250
    stage_iters = 50
    mae_thresh = 3e-5  # your desired threshold

    active_mask = torch.ones((n, n), device = E.device, dtype = torch.bool)
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

    return E


def marg_consistency_diagnostics(V_compress, E, n):

    # === Marginal Consistency Diagnostics ===
    row_margs = E.sum(dim=-1)  # [n, n, l]
    col_margs = E.sum(dim=-2)  # [n, n, l]
    V_i = V_compress[:, None, :]  # [1, n, 1, l]
    V_j = V_compress[None, :, :]  # [1, 1, n, l]

    row_diff = torch.abs(row_margs - V_i)
    col_diff = torch.abs(col_margs - V_j)
    diag_mask = 1 - torch.eye(n, device=E.device).unsqueeze(-1)  # [n, n, 1] ??
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

