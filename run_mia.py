import os
# numpy can only use at most 1 threads (we are running multiple jobs in parallel)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
from private_response import PrivateResponseModel
import numpy as np
from utils import load_dataset_info
import json

all_datasets = ['cifar10']
all_neg_log_b_vals = [12, 16, 20, 24, 28, 32]
num_trials = 250 # run 4 times for 1000 trials
max_T = 1000000
record_interval = 50000
all_recorded_T = list(range(record_interval, max_T + 1, record_interval))

b_idx = int(sys.argv[1]) # 0,1,2,3,4,5
all_neg_log_b_vals = [all_neg_log_b_vals[b_idx]]

results_dict = {}
for dataset in all_datasets:
    prm = PrivateResponseModel(m=128, dataset=dataset, split='train', load_labels_only=True)
    info = load_dataset_info(dataset)
    num_train = info['num_train']

    for neg_log_b in all_neg_log_b_vals:
        b = 2 ** (-neg_log_b)
        recorded_mia_accs = {T: [None] * num_trials for T in all_recorded_T}
        
        for i in range(num_trials):
            permuted_queries = np.random.choice(num_train, size=max_T, replace=True)
            prm.reset()

            num_responses_made = 0
            for idx in permuted_queries:
                _ = prm.predict(idx=idx, b=2**(-neg_log_b), use_one_hot=True)
                num_responses_made += 1
                if num_responses_made in all_recorded_T:
                    mia_acc = prm.mia_accuracy()
                    recorded_mia_accs[num_responses_made][i] = mia_acc

        if dataset not in results_dict:
            results_dict[dataset] = {}
        if neg_log_b not in results_dict[dataset]:
            results_dict[dataset][neg_log_b] = {}
        results_dict[dataset][neg_log_b] = recorded_mia_accs

import random
suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
curr_file_dir = os.path.dirname(os.path.abspath(__file__))
results_folder_name = "final_results"
results_folder_path = os.path.join(curr_file_dir, results_folder_name)
if not os.path.exists(results_folder_path):
    os.makedirs(results_folder_path)

file_name = f"mia_{suffix}.json"
with open(os.path.join(results_folder_path, file_name), "w") as f:
    json.dump(results_dict, f)
