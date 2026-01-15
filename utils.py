from collections import Counter
import os
import tarfile
import pandas as pd
import urllib
import zipfile
import torch
import torchvision
import numpy as np
from models import resnet, wide_resnet
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import glob
from shutil import copyfile, rmtree
from datasets import load_dataset as hf_load_dataset
from torch.utils.data import default_collate
from torchvision.transforms import v2
from torch.amp import autocast, GradScaler

def load_dataset_info(dataset_name):
    if dataset_name == 'cifar10':
        return {'num_train': 50000, 'num_test': 10000, 'num_classes': 10, 'model': 'wide_resnet_28_10'}
    elif dataset_name == 'cifar100':
        return {'num_train': 50000, 'num_test': 10000, 'num_classes': 100, 'model': 'wide_resnet_28_10'}
    elif dataset_name == 'adult':
        return {'num_train': 39073, 'num_test': 9769, 'num_classes': 2, 'model': 'xgboost'}
    elif dataset_name == 'bank':
        return {'num_train': 32950, 'num_test': 8238, 'num_classes': 2, 'model': 'xgboost'}
    elif dataset_name == 'imdb':
        return {'num_train': 25000, 'num_test': 25000, 'num_classes': 2, 'model': 'bert_small'}
    elif dataset_name == 'ag_news':
        return {'num_train': 120000, 'num_test': 7600, 'num_classes': 4, 'model': 'bert_small'}

def load_ground_truth(dataset_name, train): # get the ground truth as a flat numpy array of int
    if dataset_name == 'adult':
        _, _, y_train, y_test = load_adult()
        if train:
            return y_train.to_numpy()
        else:
            return y_test.to_numpy()
    elif dataset_name == 'bank':
        _, _, y_train, y_test = load_bank()
        if train:
            return y_train.to_numpy()
        else:
            return y_test.to_numpy()
    elif dataset_name == 'cifar10':
        if train:
            data = torchvision.datasets.CIFAR10(root='data', train=True, download=True)
            return np.array(data.targets)
        else:
            data = torchvision.datasets.CIFAR10(root='data', train=False, download=True)
            return np.array(data.targets)
    elif dataset_name == 'cifar100':
        if train:
            data = torchvision.datasets.CIFAR100(root='data', train=True, download=True)
            return np.array(data.targets)
        else:
            data = torchvision.datasets.CIFAR100(root='data', train=False, download=True)
            return np.array(data.targets)
    elif dataset_name == 'imdb':
        dataset = hf_load_dataset("imdb")
        if train:
            return np.array(dataset['train']['label'])
        else:
            return np.array(dataset['test']['label'])
    elif dataset_name == 'ag_news':
        dataset = hf_load_dataset("ag_news")
        if train:
            return np.array(dataset['train']['label'])
        else:
            return np.array(dataset['test']['label'])

def load_adult():
    # we load UCI adult and UCI credit
    if not os.path.exists("./data/adult"):
        urllib.request.urlretrieve("https://archive.ics.uci.edu/static/public/2/adult.zip", "adult.zip")
        with zipfile.ZipFile("adult.zip", 'r') as zip_ref:
            zip_ref.extractall("./data/adult")
        os.remove("adult.zip")

    train_data = pd.read_csv("./data/adult/adult.data", header=None, delimiter=', ', engine='python')
    test_data = pd.read_csv("./data/adult/adult.test", header=None, skiprows=1, delimiter=', ', engine='python')
    full_data = pd.concat([train_data, test_data], ignore_index=True)
    full_data = full_data.replace('?', pd.NA)
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                    'hours-per-week', 'native-country', 'income']
    full_data.columns = column_names
    full_data = full_data.drop('education', axis=1)
    full_data['income'] = full_data['income'].apply(lambda x: 0 if x == '<=50K' or x == '<=50K.' else 1)
    # cast all categorical columns to 'category' dtype
    for col in full_data.select_dtypes(include='object').columns:
        full_data[col] = full_data[col].astype('category')

    X = full_data.drop('income', axis=1)
    y = full_data['income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
    
def load_bank():
    if not os.path.exists("./data/bank"):
        urllib.request.urlretrieve("https://archive.ics.uci.edu/static/public/222/bank+marketing.zip", "bank.zip")
        with zipfile.ZipFile("bank.zip", 'r') as zip_ref:
            zip_ref.extractall("./data/bank")
        with zipfile.ZipFile("data/bank/bank.zip", 'r') as zip_ref:
            zip_ref.extractall("./data/bank/bank")
        with zipfile.ZipFile("data/bank/bank-additional.zip", 'r') as zip_ref:
            zip_ref.extractall("./data/bank")
        os.remove("bank.zip")
        os.remove("./data/bank/bank.zip")
        os.remove("./data/bank/bank-additional.zip")
    data = pd.read_csv("./data/bank/bank-additional/bank-additional-full.csv", sep=';')
    data = data.replace('unknown', pd.NA)
    data['y'] = data['y'].apply(lambda x: 0 if x == 'no' else 1)
    for col in data.select_dtypes(include='object').columns:
        data[col] = data[col].astype('category')
    X = data.drop('y', axis=1)
    y = data['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def load_text_dataset(dataset_name, tokenizer):
    if dataset_name.lower() == "imdb":
        dataset = hf_load_dataset("imdb")
        text_column = "text"
        test_split = "test"
        max_length = 512
        num_classes = 2
    elif dataset_name.lower() == "ag_news":
        dataset = hf_load_dataset("ag_news")
        text_column = "text"
        test_split = "test"
        max_length = 512
        num_classes = 4
    
    def tokenize(batch):
        return tokenizer(batch[text_column], truncation=True,
                         padding=False, # we will pad later with DataCollator
                         max_length=max_length,
                         )
    
    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.remove_columns(
        [c for c in dataset['train'].column_names if c not in ("input_ids", "attention_mask", "label")]
    )
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format("torch")

    return dataset, test_split, num_classes

# this function downloads and builds the cinic10-imagenet dataset and put it in ./data/cinic10-imagenet
def build_cinic10_imagenet():
    cinic_directory = "./data/cinic-10"
    imagenet_directory = "./data/cinic-10-imagenet"
    combine_train_valid = True
    symlink = False # we delete the cinic 10 directory completely after processing, so no need to symlink
    urllib.request.urlretrieve("https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz", "cinic10.tar.gz")
    with tarfile.open("cinic10.tar.gz", "r:gz") as tar:
        tar.extractall(path=cinic_directory)
    os.remove("cinic10.tar.gz")
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    sets = ['train', 'valid', 'test']
    if not os.path.exists(imagenet_directory):
        os.makedirs(imagenet_directory)
    if not os.path.exists(imagenet_directory + '/train'):
        os.makedirs(imagenet_directory + '/train')
    if not os.path.exists(imagenet_directory + '/test'):
        os.makedirs(imagenet_directory + '/test')

    for c in classes:
        if not os.path.exists('{}/train/{}'.format(imagenet_directory, c)):
            os.makedirs('{}/train/{}'.format(imagenet_directory, c))
        if not os.path.exists('{}/test/{}'.format(imagenet_directory, c)):
            os.makedirs('{}/test/{}'.format(imagenet_directory, c))
    
    for s in sets:
        for c in classes:
            source_directory = '{}/{}/{}'.format(cinic_directory, s, c)
            filenames = glob.glob('{}/*.png'.format(source_directory))
            for fn in filenames:
                dest_fn = fn.split('/')[-1]
                if (s == 'train' or s == 'valid') and combine_train_valid and 'cifar' not in fn.split('/')[-1]:
                    dest_fn = '{}/train/{}/{}'.format(imagenet_directory, c, dest_fn)
                    if symlink:
                        if not os.path.islink(dest_fn):
                            os.symlink(fn, dest_fn)
                    else:
                        copyfile(fn, dest_fn)
                    
                elif (s == 'train') and 'cifar' not in fn.split('/')[-1]:
                    dest_fn = '{}/train/{}/{}'.format(imagenet_directory, c, dest_fn)
                    if symlink:
                        if not os.path.islink(dest_fn):
                            os.symlink(fn, dest_fn)
                    else:
                        copyfile(fn, dest_fn)
                        
                elif (s == 'valid') and 'cifar' not in fn.split('/')[-1]:
                    dest_fn = '{}/valid/{}/{}'.format(imagenet_directory, c, dest_fn)
                    if symlink:
                        if not os.path.islink(dest_fn):
                            os.symlink(fn, dest_fn)
                    else:
                        copyfile(fn, dest_fn)
                        
                elif s == 'test' and 'cifar' not in fn.split('/')[-1]:
                    dest_fn = '{}/test/{}/{}'.format(imagenet_directory, c, dest_fn)
                    if symlink:
                        if not os.path.islink(dest_fn):
                            os.symlink(fn, dest_fn)
                    else:
                        copyfile(fn, dest_fn)
    # delete the cinic directory to save space
    rmtree(cinic_directory)


def load_dataset(dataset_name, train, transform, return_num_classes=False):
    if 'cifar100' in dataset_name:
        dataset = torchvision.datasets.CIFAR100(root='data', train=train, transform=transform, download=True)
        if return_num_classes:
            return dataset, 100
        return dataset
    elif 'cifar10' in dataset_name:
        dataset = torchvision.datasets.CIFAR10(root='data', train=train, transform=transform, download=True)
        if return_num_classes:
            return dataset, 10
        return dataset
    else:
        raise ValueError(f"Unknown dataset {dataset_name}, should be cifar10 or cifar100")
    
def load_cinic10_imagenet(train, transform, return_num_classes=False):
    if not os.path.exists("./data/cinic-10-imagenet"):
        build_cinic10_imagenet()
    split = 'train' if train else 'test'
    dataset = torchvision.datasets.ImageFolder(root='./data/cinic-10-imagenet/{}'.format(split), transform=transform)
    if return_num_classes:
        return dataset, 10
    return dataset

def get_image_transform(train):
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32,4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # imagenet stats, no privacy leakage
            ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # imagenet stats, no privacy leakage
            ])
    return transform

def build_model(model_name, num_classes):
    if model_name.startswith('resnet'):
        # it should be resnet_{number} or resnet_{number}_dropout
        depth = int(model_name.split('_')[1])
        dropout_rate = 0.3 if 'dropout' in model_name else 0.0
        return resnet(depth=depth, num_classes=num_classes, dropout_rate=dropout_rate)
    elif model_name.startswith('wide_resnet'):
        # it should be wide_resnet_{number}_{number} or wide_resnet_{number}_{number}_dropout
        parts = model_name.split('_')
        depth = int(parts[2])
        width = int(parts[3])
        dropout_rate = 0.3 if 'dropout' in model_name else 0.0
        return wide_resnet(depth=depth, width=width, num_classes=num_classes, dropout_rate=dropout_rate)
    
def make_balanced_sampler(dataset, num_classes):
    labels = []
    for i in range(len(dataset)):
        labels.append(dataset[i][1])
    # make sure all entries in labels are int from 0 to num_classes-1
    assert all(isinstance(label, int) and 0 <= label < num_classes for label in labels), "Labels should be int from 0 to num_classes-1"
    label_counts = Counter(labels)
    weights = torch.tensor(
        [1.0 / label_counts[label] for label in labels],
        dtype=torch.double,
    )
    return torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
    
def train_model(model_name, train_data, num_classes=10, amp=False, batch_size=128, balanced_sampler=False, cutmix_mixup=False):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if balanced_sampler:
        sampler = make_balanced_sampler(train_data, num_classes)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    if cutmix_mixup:
        cutmix = v2.CutMix(num_classes=num_classes)
        mixup = v2.MixUp(num_classes=num_classes)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

        def collate_fn(batch):
            return cutmix_or_mixup(*default_collate(batch))
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=4, collate_fn=collate_fn, sampler=sampler)

    else:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, num_workers=4, sampler=sampler)

    model = build_model(model_name, num_classes).to(device)
    loss = torch.nn.CrossEntropyLoss()
    lr = 0.1 * batch_size / 128 # linear scaling rule
    if model_name.startswith('wide_resnet'):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    elif model_name.startswith('resnet'):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
    num_epochs = 200

    if amp:
        scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            if amp:
                with autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(x)
                    loss_value = loss(outputs, y)
                scaler.scale(loss_value).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(x)
                loss_value = loss(outputs, y)
                loss_value.backward()
                optimizer.step()

        lr_scheduler.step()

    return model

@torch.no_grad()
def test_model(model, test_data, return_type='logit', return_acc=False, batch_size=500, amp=False):
    # return type can be: logit or score or pred
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    model.eval()
    # return logits on all test data: num_test_examples x num_classes
    all_outputs = []
    correct = 0
    total = 0

    for images, labels in test_loader:
        images = images.to(device)
        if amp:
            with autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
        else:
            outputs = model(images)
        outputs = outputs.float() # in case model outputs fp16, cast to float32 before softmax
        if return_acc:
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum().item()
        # all_logits.append(outputs.cpu())
        if return_type == 'logit':
            all_outputs.append(outputs.cpu())
        elif return_type == 'score':
            scores = torch.nn.functional.softmax(outputs, dim=1)
            all_outputs.append(scores.cpu())
        elif return_type == 'pred':
            if return_acc:
                all_outputs.append(predicted.cpu().unsqueeze(1)) # make it num_examples x 1
            else:
                _, preds = torch.max(outputs, 1)
                all_outputs.append(preds.cpu().unsqueeze(1)) # make it num_examples x 1

    all_outputs = torch.cat(all_outputs, dim=0).numpy() # num_test_examples x num_classes or num_test_examples x 1
    # flat if pred
    if return_type == 'pred':
        all_outputs = all_outputs.flatten() # num_test_examples
    
    if return_acc:
        return all_outputs, correct / total
    return all_outputs
