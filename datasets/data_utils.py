import numpy as np
from torch.utils.data import Dataset

from datasets.dataset import get_anomalyncd_datasets

from copy import deepcopy
import os
import torch


def get_datasets(train_transform, test_transform, args):
    """
    Get datasets for training and testing
    
    Args:
        train_transform: transform to apply to training data
        test_transform: transform to apply to test data
        args: arguments from the command line
    Return: 
        train_dataset: MergedDataset which concatenates labelled and unlabelled
        test_dataset: unlabelled for testing
        unlabelled_train_examples_test: unlabelled for training
    """

    # Get datasets
    datasets = get_anomalyncd_datasets(train_transform=train_transform, test_transform=test_transform,
                            base_path=args.base_data_path,
                            labelled_classes=args.train_classes,
                            unlabelled_classes=args.unlabeled_classes,
                            category=args.category,
                            data_root=args.crop_data_path)
    

    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    for _, dataset in datasets.items():
        if dataset is not None:
            dataset.target_transform = target_transform

    # Train split (labelled and unlabelled classes) for training
    train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                unlabelled_dataset=deepcopy(datasets['train_unlabelled']))

    test_dataset = datasets['test']

    return train_dataset, test_dataset


def get_class_splits(args):
    """
    Get the anomaly class names for the labelled and unlabelled category
    """

    # base_category
    base_path = os.path.join(args.base_data_path, 'images')
    args.train_classes = sorted(os.listdir(base_path))
    base_num = 0
    for base_class in args.train_classes:
        base_class_path = os.path.join(base_path, base_class)
        base_class_files = sorted(os.listdir(base_class_path))
        base_num += len(base_class_files)

    # novel_category
    novel_path = os.path.join(args.crop_data_path, args.category, 'images')
    args.unlabeled_classes = sorted(os.listdir(novel_path))    
    novel_num = 0
    for novel_class in args.unlabeled_classes:
        novel_class_path = os.path.join(novel_path, novel_class)
        novel_class_files = sorted(os.listdir(novel_class_path))
        novel_num += len(novel_class_files)        

    print(f'labeled class: {args.train_classes}')
    print(f'unlabeled class: {args.unlabeled_classes}')

    return args


class MergedDataset(Dataset):

    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """

    def __init__(self, labelled_dataset, unlabelled_dataset):

        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.target_transform = None

    def __getitem__(self, item):

        if item < len(self.labelled_dataset):
            img, label, uq_idx, image_path, mask, mask_path = self.labelled_dataset[item]

        else:

            img, label, uq_idx, image_path, mask, mask_path = self.unlabelled_dataset[item - len(self.labelled_dataset)]


        return img, label, image_path, mask, mask_path

    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)
    

class DistributedWeightedSampler(torch.utils.data.distributed.DistributedSampler):
    """
    sample elements from a given list of indices with given probabilities.
    """
    def __init__(self, dataset, weights, num_samples, num_replicas=None, rank=None,
                 replacement=True, generator=None):
        super(DistributedWeightedSampler, self).__init__(dataset, num_replicas, rank)
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator
        self.weights = self.weights[self.rank::self.num_replicas]
        self.num_samples = self.num_samples // self.num_replicas

    def __iter__(self):
        rand_tensor = torch.multinomial(self.weights, self.num_samples, self.replacement, generator=self.generator)
        rand_tensor =  self.rank + rand_tensor * self.num_replicas
        yield from iter(rand_tensor.tolist())

    def __len__(self):
        return self.num_samples


def get_pseudo_label_weights(image_path, anomaly_thred, base_category, anomaly_score_json):
    """
    Get the weights for the pseudo labels correction.

    image_path: list of full paths for sub-images (both base(AeBAD) + novel(MVTec) mixed)
    base_category: e.g. 'AeBAD_crop'
    anomaly_score_json: scores json for *novel* sub-images
    """
    sample_weights = []
    # 1이면 labeled(AeBAD), 0이면 unlabeled(MVTec)
    mask_lab = torch.ones(len(image_path))

    for i, pth in enumerate(image_path):
        # 윈도우 경로를 리눅스 스타일로 통일
        norm_pth = pth.replace("\\", "/")

        # base_category(AeBAD_crop)가 경로에 없으면 => unlabeled (MVTec)
        if base_category not in norm_pth:
            mask_lab[i] = 0

            # pseudo label weight를 쓰지 않는 경우
            if anomaly_thred == -1:
                sample_weights.append(0.0)
            else:
                parts = norm_pth.split("/")
                # 예) data/mvtec_musc_crop/bottle/images/broken_large/000_crop0.png
                # parts = ["data", "mvtec_musc_crop", "bottle", "images", "broken_large", "000_crop0.png"]
                ano_type = parts[-2]   # "broken_large"
                filename = parts[-1]   # "000_crop0.png"

                # json에 키가 없으면 그냥 0으로 처리해서 크래시 안 나게
                try:
                    ano_score = anomaly_score_json[ano_type][filename]
                except KeyError:
                    ano_score = 0.0

                # anomaly_thred - ano_score 가 클수록 더 신뢰도 높은 pseudo label
                sample_weights.append(max(0.0, anomaly_thred - ano_score))

        # base_category가 포함된 경로(AeBAD_crop)는 labeled 샘플 → sample_weights에 안 넣음

    # sample_weights 길이 = "언라벨 샘플 수"
    # MGRL에서 sample_weights * 2 해서 뷰 두 개에 맞춰줌
    return sample_weights, mask_lab