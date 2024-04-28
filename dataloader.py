from torch.utils.data import DataLoader, Sampler, Dataset
from torch.nn.utils.rnn import pad_sequence
import random
import torch
import math
# GPT
EOS_ID = 50256
# Bert
SEP_ID = 102
PAD_ID= 0
# T5
PAD_ID_T5 = 0
SEP_ID_T5 = 1

class Feature:
    def __init__(self, bert_ids, gpt2_ids, raw_text, cond=None):
        self.input_ids_bert = bert_ids
        self.input_ids_dec = [EOS_ID] + gpt2_ids
        self.lm_labels = gpt2_ids + [EOS_ID]
        if cond is not None:
            self.cond = cond

class FeatureDataset(Dataset):
    """ pytorch dataset for GPT2 training """

    def __init__(self, features):
        self.features = features

    def __getitem__(self, i):
        feat_dict = self.features[i]
        feat = Feature(**feat_dict)
        return feat

    def __len__(self):
        return len(self.features)

    @staticmethod
    def collate(features):
        input_ids_bert = pad_sequence([torch.tensor(f.input_ids_bert)
                                  for f in features],
                                 batch_first=True, padding_value=0)
        input_ids_dec = pad_sequence([torch.tensor(f.input_ids_dec, dtype=torch.long)
                               for f in features],
                              batch_first=True, padding_value=0)
        lm_labels = pad_sequence([torch.tensor(f.lm_labels, dtype=torch.long)
                               for f in features],
                              batch_first=True, padding_value=-1)
        if not hasattr(features[0], 'cond'):
            cond = [None for f in features]
        else:
            if isinstance(features[0].cond, int) or isinstance(features[0].cond, str):
                cond = [f.cond for f in features]
            else: #cont feature
                cond = pad_sequence([torch.tensor(f.cond)
                               for f in features],
                              batch_first=True, padding_value=0)

        return (input_ids_bert, input_ids_dec, lm_labels, cond)

class BucketSampler(Sampler):
    """
    this sampler will sort data by sequence length
    """
    def __init__(self, lens, bucket_size, batch_size,
                 droplast=False, shuffle=True):
        self._lens = lens
        self._batch_size = batch_size
        self._bucket_size = bucket_size
        self._droplast = droplast
        self._shuf = shuffle

    def __iter__(self):
        ids = list(range(len(self._lens)))
        if self._shuf:
            random.shuffle(ids)
        buckets = [sorted(ids[i:i+self._bucket_size],
                          key=lambda i: self._lens[i], reverse=True)
                   for i in range(0, len(ids), self._bucket_size)]
        batches = [bucket[i:i+self._batch_size]
                   for bucket in buckets
                   for i in range(0, len(bucket), self._batch_size)]
        if self._droplast:
            batches = [batch for batch in batches
                       if len(batch) == self._batch_size]
        if self._shuf:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        bucket_sizes = ([self._bucket_size]
                        * (len(self._lens) // self._bucket_size)
                        + [len(self._lens) % self._bucket_size])
        if self._droplast:
            return sum(s//self._batch_size for s in bucket_sizes)
        else:
            return sum(math.ceil(s/self._batch_size) for s in bucket_sizes)

def get_data(dataset):
    if dataset == 'summary':
        pass
    elif dataset == 'commonsense':
        pass
    else:
        pass

from datasets import load_dataset
from torch.utils.data import DataLoader

def load_winogrande(subset="winogrande_debiased"):
    # Load the WinoGrande dataset
    dataset = load_dataset("winogrande", subset)

    # Split the dataset into training and testing sets
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # Create data loaders for training and testing sets
    return train_dataset, test_dataset

def load_gsm8k():
    # Load the dataset from Hugging Face
    dataset = load_dataset("gsm8k", "main")

    # Get the training and testing splits
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    return train_dataset, test_dataset

def load_tripadvisor(train_ratio=0.9):
    dataset = load_dataset("argilla/tripadvisor-hotel-reviews")
    
    # Get the training dataset
    train_dataset = dataset["train"]

    # Split the training dataset into train and test
    train_size = int(train_ratio * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

    return train_dataset, test_dataset

