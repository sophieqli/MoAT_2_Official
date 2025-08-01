
import os

dsets = [
    "nltcs", "kdd", "plants", "baudio", "jester",
    "bnetflix", "accidents", "tretail", "pumsb_star", "dna",
    "kosarek", "msweb", "book", "tmovie", "cwebkb", "cr52",
    "c20ng", "bbc", "ad", "msnbc"
]


def to_base4_from_2bit_chunks(bits):
    # If odd number of bits, pad with one 0
    if len(bits) % 2 != 0:
        bits.append(0)
    digits = []
    for i in range(0, len(bits), 2):
        digit = (bits[i] << 1) | bits[i+1]
        digits.append(str(digit))
    return digits

def dataset_base4(ds_name, tvt = "train"):
    # Read and process
    #input_path = "datasets/samp/samp.train.data"
    input_path = f"datasets/{ds_name}/{ds_name}.{tvt}.data"
    output_lines = []

    with open(input_path, "r") as file:
        for line in file:
            # Get list of integers: 0s and 1s
            bits = list(map(int, line.strip().split(',')))
            digits = to_base4_from_2bit_chunks(bits)
            output_lines.append(",".join(digits))

    # Write to output
    out_dir = f"datasets/{ds_name}4"
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"{ds_name}4.{tvt}.data")
    print("Writing to:", os.path.abspath(out_path))

    with open(out_path, "w") as outfile:
        outfile.write("\n".join(output_lines))

for i in dsets: 
	dataset_base4(i, "train")
	dataset_base4(i, "valid")
	dataset_base4(i, "test")
