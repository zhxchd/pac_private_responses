import os
import numpy as np
import json
from utils import load_dataset_info
from pac_privacy_utils import get_noise_components, update_p, posterior_success_guarantee, is_confident

class PrivateResponseModel:
    def __init__(self, m, dataset, split, model=None, load_labels_only=False):
        self.m = m
        self.__s = np.random.randint(0, m)
        self.dataset = dataset
        self.p = np.zeros(m) + 1.0 / m
        self.B = 0.0
        self.num_queries = 0
        self.split = split

        # split could be 'train' or 'test' or 'cinic'

        dataset_info = load_dataset_info(dataset)
        num_train = dataset_info['num_train']
        model = dataset_info['model'] if model is None else model # model can override default
        self.d = dataset_info['num_classes']

        # load models and model predictions on test set
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(curr_dir, 'saved_models', dataset)
        self.scores = []
        self.labels = []
        self.members = np.zeros((m, num_train), dtype=bool)

        # load scores and membership info, s.t. we don't have to perform inference every time
        for i in range(m):
            score_path = os.path.join(base_dir, f'X_{i}', f'{model}_{split}_scores.npy')
            label_path = os.path.join(base_dir, f'X_{i}', f'{model}_{split}_preds.npy')
            member_path = os.path.join(base_dir, f'X_{i}', "train_data_indices.json")

            labels_i = None

            if load_labels_only or not os.path.exists(score_path):
                # try if label_path exists
                if os.path.exists(label_path):
                    labels_i = np.load(label_path) # (num_samples,)
                else:
                    scores_i = np.load(score_path) # (num_samples, num_classes)
                    labels_i = np.argmax(scores_i, axis=1) # (num_samples,)
                    del scores_i
                self.labels.append(labels_i)
                del labels_i
            else:
                scores_i = np.load(score_path) # (num_samples, num_classes)
                self.scores.append(scores_i.astype(np.float64))
                labels_i = np.argmax(scores_i, axis=1) # (num_samples,)
                self.labels.append(labels_i)
                del scores_i
                del labels_i

            with open(member_path, 'r') as f:
                members_i = json.load(f)
            self.members[i, members_i] = True

        self.labels = np.array(self.labels) # (m, num_samples)
        if load_labels_only:
            self.scores = None
        else:
            self.scores = np.array(self.scores)

        self.split_size = self.labels.shape[1]

    
    def reset(self):
        self.__s = np.random.randint(0, self.m)
        self.p = np.zeros(self.m) + 1.0 / self.m
        self.B = 0.0
        self.num_queries = 0

    def predict(self, idx, b, use_one_hot, use_svd=True, return_confidence=False, alpha=0.05):
        if use_one_hot:
            m_model_labels = self.labels[:, idx]  # shape (m,)
            outputs = np.zeros((self.m, self.d), dtype=np.float64)  # m, num_classes
            outputs[np.arange(self.m), m_model_labels] = 1.0
        else:
            assert self.scores is not None, "Scores not loaded. Set load_labels_only=False during initialization."
            outputs = self.scores[:, idx, :]  # shape (m, num_classes)

        noise_U, noise_lambda = get_noise_components(self.p, outputs, b, use_svd=use_svd)
        noise = (np.random.randn(self.d) * np.sqrt(noise_lambda))
        if noise_U is not None:
            noise = noise_U @ noise
        priv_output = outputs[self.__s] + noise
        self.p = update_p(self.p, outputs, priv_output, noise_U, noise_lambda)
        
        assert not np.any(np.isnan(priv_output)), "Private output contains NaN values."
        assert not np.any(np.isnan(self.p)), "Probability distribution contains NaN values."

        self.B += b
        self.num_queries += 1

        if return_confidence:
            is_conf = is_confident(priv_output, noise_U, noise_lambda, alpha=alpha)
            return priv_output, is_conf

        return priv_output
    
    def get_mia_guarantee(self):
        return posterior_success_guarantee(self.B)
    
    def mia_accuracy(self):
        # using the p probabilities, to calculate the probability of each train set
        membership_probs = np.sum(self.p[:, None] * self.members, axis=0)
        ground_truth = self.members[self.__s].astype(int)
        acc = ((membership_probs >= 0.5).astype(int) == ground_truth).mean()
        return acc
