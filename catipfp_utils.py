
#funcs 


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

