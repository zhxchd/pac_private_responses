import os
# numpy can only use at most 1 threads (we are running multiple jobs in parallel)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
from private_response import PrivateResponseModel
import numpy as np
from utils import load_ground_truth, load_dataset_info
from sklearn.metrics import top_k_accuracy_score
import json

all_datasets = ['imdb', 'ag_news', 'cifar10', 'cifar100', 'adult', 'bank']
all_output_type = ['one_hot', 'score']
all_neg_log_b_vals = list(range(32, 0, -4))
num_trials = 1000

# comment the following two lines to run all combinations in one go
output_type_idx = int(sys.argv[1])  # 0 or 1
neg_log_b_idx = int(sys.argv[2])  # 0 to 7

all_output_type = [all_output_type[output_type_idx]]
all_neg_log_b_vals = [all_neg_log_b_vals[neg_log_b_idx]]

results_dict = {}
for dataset in all_datasets:
    # if all_output_type only has one_hot
    if all_output_type == ['one_hot']:
        prm = PrivateResponseModel(m=128, dataset=dataset, split='test', load_labels_only=True)
    else:
        prm = PrivateResponseModel(m=128, dataset=dataset, split='test')
    info = load_dataset_info(dataset)
    num_test = info['num_test']
    num_classes = info['num_classes']
    ground_truth = load_ground_truth(dataset, train=False)

    for output_type in all_output_type:
        use_one_hot = 'one_hot' in output_type
        for neg_log_b in all_neg_log_b_vals:
            b = 2 ** (-neg_log_b)
            top1_accs = [None] * num_trials
            top5_accs = [None] * num_trials
            for i in range(num_trials):
                prm.reset()
                permuted_queries = np.random.permutation(num_test)

                priv_outputs = np.empty((num_test, num_classes))
                for idx in permuted_queries:
                    priv_outputs[idx] = prm.predict(idx=idx, b=b, use_one_hot=use_one_hot)

                pred = np.argmax(priv_outputs, axis=1)
                top1_accs[i] = (pred == ground_truth).mean()
                if dataset == 'cifar100':
                    top5_accs[i] = top_k_accuracy_score(ground_truth, priv_outputs, k=5)

            if dataset not in results_dict:
                results_dict[dataset] = {}
            if output_type not in results_dict[dataset]:
                results_dict[dataset][output_type] = {}
            if neg_log_b not in results_dict[dataset][output_type]:
                results_dict[dataset][output_type][neg_log_b] = {}
            results_dict[dataset][output_type][neg_log_b]['top1_acc'] = top1_accs
            results_dict[dataset][output_type][neg_log_b]['top5_acc'] = top5_accs

import random
suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
curr_file_dir = os.path.dirname(os.path.abspath(__file__))
results_folder_name = "final_results"
results_folder_path = os.path.join(curr_file_dir, results_folder_name)
if not os.path.exists(results_folder_path):
    os.makedirs(results_folder_path)

file_name = f"prediction_{suffix}.json"
with open(os.path.join(results_folder_path, file_name), "w") as f:
    json.dump(results_dict, f)
