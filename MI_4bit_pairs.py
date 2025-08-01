
import networkx as nx
import numpy as np
from catipfp_train import *

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

def MIgroup(ds_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train, valid, test = load_data2("datasets/", ds_name)
    x_train = train.x.to(device)
    x_val = valid.x.to(device)
    x_test = test.x.to(device)

    x_train = pad_if_odd(x_train, device)
    x_val = pad_if_odd(x_val, device)
    x_test = pad_if_odd(x_test, device)
    print("on devices ", x_train.device, x_val.device)

    #now calculate MI matrix shape (n,n)

    num_classes = 4
    temp_moat = MoAT(n=x_train.shape[1], x=x_train, num_classes=num_classes, device=device, K=1)
    mi_matrix = torch.sigmoid(temp_moat.W.detach())
    mi_np = mi_matrix.cpu().numpy()
    #print(mi_matrix)
    print("starting to cluster")
    pairs = pair_variables_by_max_mi(mi_np)
    #print("cluster pairs ", pairs)

    print("re-ordering cols based on cluster")
    reordered_indices = [i for pair in pairs for i in pair]
    x_trainorder = x_train[:, reordered_indices]
    x_valorder = x_val[:, reordered_indices]
    x_testorder = x_test[:, reordered_indices]

    def binary_tensor_to_base4(x_bin):
        even_bits = x_bin[:, ::2]
        odd_bits = x_bin[:, 1::2]
        base4_tensor = (even_bits << 1) | odd_bits
        return base4_tensor

    x_trainoutput = binary_tensor_to_base4(x_trainorder)
    x_valoutput = binary_tensor_to_base4(x_valorder)
    x_testoutput = binary_tensor_to_base4(x_testorder)

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
MIgroup("c20ng")
'''
longaf = ["cr52", "c20ng", "bbc", "ad"]
for i in longaf:
    MIgroup(i)

'''
