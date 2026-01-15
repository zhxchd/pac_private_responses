import torch
import numpy as np

torch.manual_seed(42)
torch.cuda.manual_seed(42)

import json
import os
import argparse
from utils import load_dataset, get_image_transform, train_model, test_model, load_imagenet32, load_cinic10_imagenet

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str) # dataset to use
parser.add_argument('split_id', type=str) # split id, in the folder we have the indices of the training data
parser.add_argument('model', type=str)
args = parser.parse_args()

train_data, num_classes = load_dataset(dataset_name=args.dataset, train=True, transform=get_image_transform(train=True), return_num_classes=True)

curr_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(curr_dir, 'saved_models', args.dataset, args.split_id)
assert os.path.exists(base_dir), f"{base_dir} does not exist"

with open(os.path.join(base_dir, "train_data_indices.json"), 'r') as f:
    train_data_indices = json.load(f)

train_data_subset = torch.utils.data.Subset(train_data, train_data_indices)

model = train_model(args.model, train_data_subset, num_classes, amp=True, cutmix_mixup=True)

torch.save(model.state_dict(), os.path.join(base_dir, f"{args.model}_state_dict.pt"))

train_data_for_eval = load_dataset(dataset_name=args.dataset, train=True, transform=get_image_transform(train=False))
train_preds = test_model(model, train_data_for_eval, return_type='pred', batch_size=2000, amp=True)

test_data = load_dataset(dataset_name=args.dataset, train=False, transform=get_image_transform(train=False))
test_scores, test_acc = test_model(model, test_data, return_type='score', return_acc=True, batch_size=2000, amp=True)
print(f"Test accuracy ({args.dataset} {args.split_id} with {args.model}) : {test_acc*100}%")

imagenet_data = load_imagenet32(train=True, transform=get_image_transform(train=False))
imagenet_preds = test_model(model, imagenet_data, return_type='pred', batch_size=2000, amp=True)

if args.dataset == 'cifar10':
    cinic_train = load_cinic10_imagenet(train=True, transform=get_image_transform(train=False))
    cinic_test = load_cinic10_imagenet(train=False, transform=get_image_transform(train=False))
    cinic_preds_train = test_model(model, cinic_train, return_type='pred', batch_size=2000, amp=True)
    cinic_preds_test = test_model(model, cinic_test, return_type='pred', batch_size=2000, amp=True)
    # cinic_preds = combine them into one
    cinic_preds = np.concatenate([cinic_preds_train, cinic_preds_test])

np.save(os.path.join(base_dir, f"{args.model}_train_preds.npy"), train_preds)
np.save(os.path.join(base_dir, f"{args.model}_test_scores.npy"), test_scores)
if args.dataset == 'cifar10':
    np.save(os.path.join(base_dir, f"{args.model}_cinic_preds.npy"), cinic_preds)
