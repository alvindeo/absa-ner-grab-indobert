import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from src.config import RANDOM_STATE, TEST_SIZE, ABSA_DATASET_PATH, NER_DATASET_PATH

class ABSADataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

class NERDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def get_absa_data_splits():
    df_absa = pd.read_csv(ABSA_DATASET_PATH)
    df_absa.dropna(inplace=True)
    
    label_map = {'negatif': 0, 'netral': 1, 'positif': 2}
    df_absa['label'] = df_absa['sentiment'].map(label_map)
    
    X_train_text, X_test_text, X_train_aspect, X_test_aspect, y_train, y_test = train_test_split(
        df_absa['text'].tolist(),
        df_absa['aspect'].tolist(),
        df_absa['label'].tolist(),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_absa['label'].tolist()
    )
    
    # Random Over-Sampling untuk kelas netral (label 1) ke target 350
    ros = RandomOverSampler(sampling_strategy={1: 350}, random_state=RANDOM_STATE)
    X_train_combined = np.column_stack((X_train_text, X_train_aspect))
    X_train_res, y_train_res = ros.fit_resample(X_train_combined, y_train)
    
    X_train_text_res = X_train_res[:, 0].tolist()
    X_train_aspect_res = X_train_res[:, 1].tolist()
    
    return X_train_text_res, X_train_aspect_res, y_train_res, X_test_text, X_test_aspect, y_test

def get_ner_data_splits():
    sentences = []
    sentence_labels = []
    current_tokens = []
    current_labels = []

    with open(NER_DATASET_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "":
                if current_tokens:
                    sentences.append(current_tokens)
                    sentence_labels.append(current_labels)
                    current_tokens = []
                    current_labels = []
            else:
                parts = line.split('\t')
                if len(parts) == 2:
                    current_tokens.append(parts[0])
                    current_labels.append(parts[1])
                    
    ner_label_list = ['O', 'B-ASPECT', 'B-PROD', 'B-ORG']
    label2id = {label: i for i, label in enumerate(ner_label_list)}
    id2label = {i: label for i, label in enumerate(ner_label_list)}
    
    train_idx, test_idx = train_test_split(
        range(len(sentences)), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    train_sentences = [sentences[i] for i in train_idx]
    train_tags = [[label2id[tag] for tag in sentence_labels[i]] for i in train_idx]
    
    test_sentences = [sentences[i] for i in test_idx]
    test_tags = [[label2id[tag] for tag in sentence_labels[i]] for i in test_idx]
    
    return train_sentences, train_tags, test_sentences, test_tags, label2id, id2label
