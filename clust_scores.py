import numpy as np
import torch 

def norm_cluster_score(mi_np, pairs):
    """
    Compute normalized clustering score: fraction of MI captured within clusters.
    Score = avg_intra / (avg_intra + avg_inter), ranges from 0 to 1.
    """

    n = mi_np.shape[0]
    pair_set = set()
    for a, b in pairs:
        pair_set.add((a, b))
        pair_set.add((b, a))  # ensure symmetry

    intra_sum = 0
    inter_sum = 0
    intra_count = 0
    inter_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in pair_set:
                intra_sum += mi_np[i, j]
                intra_count += 1
            else:
                inter_sum += mi_np[i, j]
                inter_count += 1

    avg_intra = intra_sum / intra_count if intra_count > 0 else 0
    avg_inter = inter_sum / inter_count if inter_count > 0 else 0

    if avg_intra + avg_inter == 0:
        return 0.0

    score = avg_intra / (avg_intra + avg_inter)

    print(f"Intra MI avg: {avg_intra:.4f}, Inter MI avg: {avg_inter:.4f}, Normalized Score: {score:.4f}")
    return score

def cluster_score(mi_np, pairs):
    """
    Compute the ratio of intra-cluster MI to total MI.

    Args:
        mi_np (ndarray): symmetric n x n MI matrix
        pairs (list of tuple): list of cluster pairs (i,j)

    Returns:
        float: ratio between 0 and 1, higher means more MI captured inside clusters.
    """
    n = mi_np.shape[0]
    assert mi_np.shape[0] == mi_np.shape[1], "MI matrix must be square"

    # Sum intra-cluster MI (sum over all pairs inside clusters)
    intra_mi = 0.0
    for i, j in pairs:
        intra_mi += mi_np[i, j]
        intra_mi += mi_np[j, i]  # symmetrical, but included for clarity

    # Sum total MI over all variable pairs (off-diagonal upper triangle)
    total_mi = np.sum(np.triu(mi_np, k=1)) * 2  # multiply by 2 for symmetry, since we sum only upper

    # Handle corner case: total MI = 0 (avoid division by zero)
    if total_mi == 0:
        return 0.0

    score = intra_mi / total_mi
    return score

def modularity_score(mi_np, pairs):
    n = mi_np.shape[0]
    assert mi_np.shape[0] == mi_np.shape[1], "MI matrix must be square"

    # Create cluster membership array: cluster_map[node] = cluster_id
    cluster_map = np.full(n, -1, dtype=int)
    for cluster_id, (i, j) in enumerate(pairs):
        cluster_map[i] = cluster_id
        cluster_map[j] = cluster_id

    # Compute degree vector k_i = sum_j A_ij
    k = mi_np.sum(axis=1)

    # Compute total edge weight m = sum of all edge weights / 2
    m = k.sum() / 2

    if m == 0:
        return 0.0

    Q = 0.0
    # Sum over all pairs (i,j)
    for i in range(n):
        for j in range(n):
            if cluster_map[i] == cluster_map[j] and cluster_map[i] != -1:
                expected = (k[i] * k[j]) / (2 * m)
                Q += (mi_np[i, j] - expected)

    Q /= (2 * m)
    return Q

import numpy as np

def matrix_rank_stable(mi_matrix, tol=1e-8):
    u, s, vh = np.linalg.svd(mi_matrix)
    rank = np.sum(s > tol)
    return rank


'''
import numpy as np
from sklearn.metrics import mutual_info_score

def compute_contraction_score_matrix(data_tensor, mi_np):
    data = data_tensor.cpu().numpy()
    N, d = data.shape
    score_matrix = np.zeros((d, d))

    for i in range(d):
        for j in range(i + 1, d):
            Y = 2 * data[:, i] + data[:, j]  # New 4-ary variable
            gain = 0.0
            print("i, j: ", i,j)
            for k in range(d):
                if k == i or k == j:
                    continue
                mi_Y_k = mutual_info_score(Y, data[:, k])
                gain += mi_Y_k - mi_np[i, k] - mi_np[j, k]
                #print("k: ", mi_Y_k, " - ", mi_np[i, k], " - ", mi_np[j,k])
                #print("supposed to be the same ", mi_Y_k, mutual_info_score(data[:, i], data[:, k]), mutual_info_score(data[:, j], data[:, k]))
                #print("---> GAIN: ", mi_Y_k - mi_np[i, k] - mi_np[j, k]) 
            score_matrix[i, j] = gain
            score_matrix[j, i] = gain  # symmetry
    #print("final score_mat ")
    print(score_matrix)
    return score_matrix
'''



import numpy as np
from sklearn.metrics import mutual_info_score
from joblib import Parallel, delayed

def compute_pair_score(mi_np, i, j):
    d = mi_np.shape[0]
    if i == j:
        return -np.inf  # never self-pair
    mask = np.ones(d, dtype=bool)
    mask[[i, j]] = False

    diff = mi_np[i, mask] - mi_np[j, mask]
    penalty = np.sum(diff ** 2)
    lambd = 0
    score = mi_np[i, j] - lambd * penalty
    return score

def compute_contraction_score_matrix(data_tensor, mi_np, n_jobs=-1):
    print("computing contraction scores")
    """
    Vectorized and parallel version of contraction score matrix computation.
    """
    data = data_tensor.cpu().numpy()
    N, d = data.shape
    score_matrix = np.zeros((d, d))

    # Precompute MI(Y; X_k) for all i, j, k using joblib for speed
    '''
    def compute_score(i, j):
        if i == j:
            return 0.0
        Y = 2 * data[:, i] + data[:, j]  # shape (N,), values in {0,1,2,3}
        mi_Y_k = np.array([mutual_info_score(Y, data[:, k]) if k != i and k != j else 0.0 for k in range(d)])
        gain = np.sum(mi_Y_k - mi_np[i, :] - mi_np[j, :])
        gain += mi_np[i, i] + mi_np[j, j] + mi_np[i, j] + mi_np[j, i]  # correct for skipped k = i or j
        return gain
    '''

    def compute_score(mi_np, i, j):
        if i == j:
            return 0.0
        Y = 2 * data[:, i] + data[:, j]  # shape (N,), values in {0,1,2,3}
        mi_Y_k = np.array([mutual_info_score(Y, data[:, k]) if k != i and k != j else 0.0 for k in range(d)])
        gain = np.sum(mi_Y_k - (2/3) * (mi_np[i, :] - mi_np[j, :] - mi_np[i,j] ) )
        if (i == 0 and j == 1) or (i == 3 and j == 5):
            print("i,j: ", i, j)
            print(mi_Y_k)
            print(gain)
        return gain

    # Parallel computation of upper triangle
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_score)(mi_np, i, j) for i in range(d) for j in range(i + 1, d)
    )

    # Fill symmetric matrix
    idx = 0
    for i in range(d):
        for j in range(i + 1, d):
            score = results[idx]
            score_matrix[i, j] = score
            score_matrix[j, i] = score
            idx += 1

    return score_matrix

