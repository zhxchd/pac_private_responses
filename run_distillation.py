import os
# numpy can only use at most 4 threads (we are running multiple jobs in parallel)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import torch

import argparse
from private_response import PrivateResponseModel
import numpy as np
from utils import load_cinic10_imagenet, get_image_transform, train_model, load_dataset, test_model
from pac_privacy_utils import posterior_success_guarantee, posterior_success_rate_to_epsilon
import json

argparser = argparse.ArgumentParser()
argparser.add_argument('--alpha', type=float, default=0.01)
args = argparser.parse_args()
alpha = args.alpha

prm = PrivateResponseModel(m=128, dataset='cifar10', split='cinic', load_labels_only=True)

total_num_queries = prm.split_size
permuted_queries = np.random.permutation(total_num_queries)

priv_labels = np.empty(total_num_queries, dtype=int)
confident_subset_indices = []
for i in permuted_queries:
    pred, is_confident = prm.predict(i, 2**-32, use_one_hot=True, return_confidence=True, alpha=alpha)
    priv_labels[i] = np.argmax(pred).item()
    if is_confident:
        confident_subset_indices.append(i)

cinic_imagenet_train_data = load_cinic10_imagenet(train=True, transform=get_image_transform(train=True))
n_train = len(cinic_imagenet_train_data)
cinic_imagenet_test_data = load_cinic10_imagenet(train=False, transform=get_image_transform(train=True))
n_test = len(cinic_imagenet_test_data)

ground_truth = np.array(cinic_imagenet_train_data.targets + cinic_imagenet_test_data.targets)
acc = (ground_truth==priv_labels).mean().item()
print(f"Overall accuracy of the 210k private responses: {acc*100:.2f}%")
print(f"{len(confident_subset_indices)} are confident ({len(confident_subset_indices)/total_num_queries*100:.2f}%)")
confident_acc = (ground_truth[confident_subset_indices]==priv_labels[confident_subset_indices]).mean().item()
print(f"Accuracy on confident subset: {confident_acc*100:.2f}%")

# Now, we relabel the CINIC-10 dataset with the private labels
cinic_imagenet_train_data.targets = priv_labels[:n_train].tolist()
cinic_imagenet_train_data.samples = [
    (path, int(label))
    for (path, _), label in zip(cinic_imagenet_train_data.samples, priv_labels[:n_train])
]

cinic_imagenet_test_data.targets = priv_labels[n_train: n_train + n_test].tolist()
cinic_imagenet_test_data.samples = [
    (path, int(label))
    for (path, _), label in zip(cinic_imagenet_test_data.samples, priv_labels[n_train: n_train + n_test])
]

relabeled_full_data = torch.utils.data.ConcatDataset([cinic_imagenet_train_data, cinic_imagenet_test_data])

num_responses = [5000, 10000, 20000, 40000, 80000, 160000, 210000]
num_responses = list(reversed(num_responses))

cifar10_test_data = load_dataset('cifar10', train=False, transform=get_image_transform(train=False))

results_dict = []
distill_model_name = 'wide_resnet_28_10_cutmix_mixup'

for n in num_responses:
    first_n_indices = permuted_queries[:n]
    first_n_confident_indices = [idx for idx in first_n_indices if idx in confident_subset_indices]
    effective_n = len(first_n_confident_indices)
    relabeled_subset = torch.utils.data.Subset(relabeled_full_data, first_n_confident_indices)
    print(f"Current number of private responses used: {n}")
    print(f"Number of confident private responses used: {effective_n}")
    model = train_model(distill_model_name, relabeled_subset, num_classes=10, amp=True, balanced_sampler=True, cutmix_mixup=True)
    _, cifar10_test_acc = test_model(model, cifar10_test_data, return_acc=True)
    print(f"CIFAR-10 test accuracy after distillation with {effective_n} confident labels from {n} private responses: {cifar10_test_acc*100}%")
    results_dict.append({
        'dataset': 'cifar10',
        'alpha': alpha,
        'accuracy_all': acc,
        'accuracy_confident_all': confident_acc,
        'num_private_responses': n,
        'num_confident_responses': effective_n,
        'cifar10_test_accuracy': cifar10_test_acc,
        'mi_bound': 2**-32 * n,
        'mia_bound': posterior_success_guarantee(2**-32 * n),
        'epsilon': posterior_success_rate_to_epsilon(posterior_success_guarantee(2**-32 * n))
    })

import random
suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=6))

curr_file_dir = os.path.dirname(os.path.abspath(__file__))
results_folder_name = "final_results"
results_folder_path = os.path.join(curr_file_dir, results_folder_name)
if not os.path.exists(results_folder_path):
    os.makedirs(results_folder_path)
with open(os.path.join(results_folder_path, f'distill_{suffix}.json'), 'w') as f:
    json.dump(results_dict, f, indent=4)
