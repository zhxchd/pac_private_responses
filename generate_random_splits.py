import os
import pathlib
import argparse
import json
import numpy as np
from utils import load_dataset_info

argparser = argparse.ArgumentParser()
argparser.add_argument('dataset', type=str) # dataset to use
argparser.add_argument('--num_splits', type=int, default=128) # number of splits to create
args = argparser.parse_args()

assert args.num_splits % 2 == 0, "num_splits must be even"

def random_subsets(n, m):
    """
    Random subset construction:
    Each element is added to m/2 randomly selected subsets.

    Args:
        n (int): universe size
        m (int): number of subsets (must be even)
        seed (int): optional seed for reproducibility

    Returns:
        List[Set[int]]: list of m subsets
    """
    if m % 2 != 0:
        raise ValueError("m must be even.")

    half_m = m // 2
    subsets = [[] for _ in range(m)]

    for element in range(n):
        sampled = np.random.choice(m, half_m, replace=False)
        for subset_id in sampled:
            subsets[subset_id].append(element)

    return subsets

info = load_dataset_info(args.dataset)
num_examples = info['num_train']

subsets = random_subsets(num_examples, args.num_splits)

curr_dir = pathlib.Path(__file__).parent.resolve()

for i in range(args.num_splits):
    base_dir = os.path.join(curr_dir, 'saved_models', args.dataset, f'X_{i}')
    os.makedirs(base_dir, exist_ok=True)
    with open(os.path.join(base_dir, 'train_data_indices.json'), 'w') as f:
        json.dump(subsets[i], f, indent=None, separators=(',', ':')) # more compact representation

# write a full list to saved_models/dataset/U
full_set = list(range(num_examples))
u_dir = os.path.join(curr_dir, 'saved_models', args.dataset, 'U')
os.makedirs(u_dir, exist_ok=True)
with open(os.path.join(u_dir, 'train_data_indices.json'), 'w') as f:
    json.dump(full_set, f, indent=None, separators=(',', ':'))
    