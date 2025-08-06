
import networkx as nx
import numpy as np
from catipfp_train import *

from scipy.stats import entropy
from numpy.linalg import eigvalsh
from clust_scores import *


dsets = [
    "nltcs", "kdd", "plants", "baudio", "jester",
    "bnetflix", "accidents", "tretail", "pumsb_star", "dna",
    "kosarek", "msweb", "book", "tmovie", "cwebkb", "cr52",
    "c20ng", "bbc", "ad", "msnbc"
]


def pair_variables_by_max_mi(MI_matrix):
    n = MI_matrix.shape[0]
    assert n % 2 == 0, "n must be even"

    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=MI_matrix[i, j])

    matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True)
    return list(matching)  # list of pairs (i,j)

def pad_if_odd(x, device):
    if x.shape[1] % 2 == 1:
        zeroscol = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=device)
        x = torch.cat([x, zeroscol], dim=1).clone()
    return x

def further_MI_analysis(mi_np):
    #normalize 
    mi_max = mi_np.max()
    if mi_max > 0:
        mi_np /= mi_max

    print(mi_np)
    n = mi_np.shape[0]


    # Sparsity / density of MI above threshold (e.g., >0.1)
    threshold = 0.1
    upper_tri_indices = np.triu_indices(n, k=1)
    mi_upper = mi_np[upper_tri_indices]
    density = np.sum(mi_upper > threshold) / len(mi_upper)
    print(f"\nDensity of MI > {threshold}: {density:.4f}")

    # Histogram entropy of MI values
    hist, bin_edges = np.histogram(mi_upper, bins=50, density=True)
    hist = hist + 1e-12  # avoid zeros for entropy
    hist /= hist.sum()  # normalize
    mi_entropy = entropy(hist)
    print(f"Entropy of MI histogram: {mi_entropy:.4f}")

    # Average MI per variable (row mean)
    avg_mi_per_var = mi_np.mean(axis=1)
    print(f"Average MI per variable (first 10): {avg_mi_per_var[:10]}")

    # Spectral properties (eigenvalues)
    eigvals = eigvalsh(mi_np)
    print(f"Eigenvalue summary:")
    print(f"  Min: {eigvals.min():.4f}")
    print(f"  Max: {eigvals.max():.4f}")
    print(f"  Mean: {eigvals.mean():.4f}")
    print(f"  Top 5: {eigvals[-5:]}")

def lostMI(mi_np, pairs, ds_name):
    #mi_np is the orig mi matrix
    #pairs is clustering 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train, valid, test = load_data2("datasets/", f"{ds_name}4")
    x_train4 = train.x.to(device)
    x_val4 = valid.x.to(device)
    x_test4 = test.x.to(device)

    tmp4 = MoAT(n=x_train4.shape[1], x=x_train4, num_classes=4, device=device, K=1)
    mi_mat4 = tmp4.MI_unorm.detach()
    
    m = len(pairs)
    lost_tot = 0 
    for i in range(m):
        for j in range(i+1, m):
            a1, a2 = pairs[i]
            b1, b2 = pairs[j]
            lost_ij = mi_np[a1, b1] + mi_np[a1, b2] + mi_np[a2, b1] + mi_np[a2, b2] - mi_mat4[i,j]
            lost_tot += lost_ij

    # Intra-group MI: sum of MI within each pair (i.e., a1 <-> a2)
    intra_mi = sum([mi_np[a1, a2] for (a1, a2) in pairs])

    # Inter-group MI: sum of MI between groups (i != j)
    inter_mi = 0
    m = len(pairs)
    for i in range(m):
        for j in range(i + 1, m):
            a1, a2 = pairs[i]
            b1, b2 = pairs[j]
            inter_mi += (
                mi_np[a1, b1] + mi_np[a1, b2] +
                mi_np[a2, b1] + mi_np[a2, b2]
            )

    #intra_mi_new = sum([mi_mat4[i,j] for i in range(m) for j in range(i, m)])
            
    print(" -> -> -> LOST MI: ", lost_tot, " INTRA_MI: ", intra_mi, " INTER_MI: ", inter_mi)
    return lost_tot, intra_mi, inter_mi



def MIgroup(ds_name):
    print("operating on ", ds_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train, valid, test = load_data2("datasets/", ds_name)
    x_train = train.x.to(device)
    x_val = valid.x.to(device)
    x_test = test.x.to(device)

    x_train = pad_if_odd(x_train, device)
    x_val = pad_if_odd(x_val, device)
    x_test = pad_if_odd(x_test, device)

    #now calculate MI matrix shape (n,n)

    num_classes = 2
    temp_moat = MoAT(n=x_train.shape[1], x=x_train, num_classes=num_classes, device=device, K=1)
    #mi_matrix = torch.sigmoid(temp_moat.W.detach())
    mi_matrix = temp_moat.MI_unorm.detach()
    #print("Any NaNs in MI matrix? TEMP MOAT DETACH ", torch.isnan(temp_moat.W.detach()).any().item())
    # print("Any NaNs in MI matrix? UGH", torch.isnan(mi_matrix).any().item())

    mi_np = mi_matrix.cpu().numpy()
    #print(mi_matrix)
    print("starting to cluster")
    
    clus_mat = compute_contraction_score_matrix(x_train, mi_np)
    pairs = pair_variables_by_max_mi(clus_mat)
    pair_scores = [compute_pair_score(mi_np, i,j) for i,j in pairs]
    print("cluster pairs ", pairs)
    print("pair scores", pair_scores)

    #NOW TAKE TOP K 
    k = 5
    top_k_indices = np.argsort(pair_scores)[-k:][::-1]
    top_k_pairs = [pairs[i] for i in top_k_indices]
    top_k_scores = [pair_scores[i] for i in top_k_indices]
    print("top k pairs ", top_k_pairs)
    print("top k scores ", top_k_scores)

    print("re-ordering cols based on cluster")
    #reordered_indices = [i for pair in pairs for i in pair]
    reordered_indices = [i for pair in top_k_pairs for i in pair]
    p_set = set(reordered_indices)
    for i in range(mi_np.shape[0]):
        if i not in p_set: reordered_indices.append(i)
    print("reordered inds")
    print(reordered_indices)

    x_trainorder = x_train[:, reordered_indices]
    x_valorder = x_val[:, reordered_indices]
    x_testorder = x_test[:, reordered_indices]

    def binary_tensor_to_base4(x_bin, top_k = -1):
        if top_k == -1:
            even_bits = x_bin[:, ::2]
            odd_bits = x_bin[:, 1::2]
            base4_tensor = (even_bits << 1) | odd_bits
            return base4_tensor
        else: 
            #only do that for the first top_k pairs!! so like if k = 3 then merge (0,1), (2,3) (4,5) bits and leave rest of cols 
            even_bits = x_bin[:, 0: 2*top_k: 2]
            odd_bits = x_bin[:, 1: 2*top_k+1: 2]
            merged = (even_bits << 1) | odd_bits

            remaining = x_bin[:, 2*top_k:]
            return torch.cat([merged, remaining], dim=1)

    x_trainoutput = binary_tensor_to_base4(x_trainorder, k)
    x_valoutput = binary_tensor_to_base4(x_valorder, k)
    x_testoutput = binary_tensor_to_base4(x_testorder, k)

    # Write to output
    out_dir = f"datasets/{ds_name}4"
    os.makedirs(out_dir, exist_ok=True)

    # Helper function to write to file
    def write_output(tensor, filename):
        out_path = os.path.join(out_dir, filename)
        print("Writing to:", os.path.abspath(out_path))
        with open(out_path, "w") as f:
            for row in tensor:
                line = ",".join(str(val.item()) for val in row)
                f.write(line + "\n")
    # Write each split
    write_output(x_trainoutput, f"{ds_name}4.train.data")
    write_output(x_valoutput, f"{ds_name}4.valid.data")
    write_output(x_testoutput, f"{ds_name}4.test.data")


#######
#Execute funct 
#######
#MIgroup("c20ng")
#MIgroup("bbc")
worse = ["plants", "baudio", "jester", "bnetflix", "tmovie", "bbc"]
same = ["nltcs", "tretail", "kosarek", "msweb", "kdd", "book", "cr52"]
better = ["msnbc", "accidents", "pumsb_star", "dna", "ad"]

MIgroup("baudio") #39.43

