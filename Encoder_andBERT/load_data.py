import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel

german_sen_emb = np.load("./data/german_sen_emb.npy")
socc_sen_emb = np.load("./data/socc_sen_emb.npy")
german_word_emb = np.load("./data/german_word_emb.npy")
socc_word_emb = np.load("./data/socc_word_emb.npy")
german_label_expert1 = np.load("./data/german_label_expert1.npy")
german_label_expert2 = np.load("./data/german_label_expert2.npy")
socc_label = np.load("./data/socc_label.npy")


def load_splited_data(training_size):
    (
        X_train_german_sen, X_test_german_sen,
        X_train_german_word, X_test_german_word,
        y_train_german1, y_test_german1,
        y_train_german2, y_test_german2
    ) = train_test_split(
        german_sen_emb, german_word_emb,
        german_label_expert1, german_label_expert2,
        train_size=training_size, random_state=42
    )
    (
        X_train_socc_sen, X_test_socc_sen,
        X_train_socc_word, X_test_socc_word,
        y_train_socc, y_test_socc,
    ) = train_test_split(
        socc_sen_emb, socc_word_emb, socc_label,
        train_size=training_size, random_state=42)
    return (
        X_train_german_sen, X_test_german_sen,
        X_train_german_word, X_test_german_word,
        y_train_german1, y_test_german1,
        y_train_german2, y_test_german2,
        X_train_socc_sen, X_test_socc_sen,
        X_train_socc_word, X_test_socc_word,
        y_train_socc, y_test_socc
    )


class SentenceDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def bert_dataloader(sentences, labels, batchsize):
    # Create train and test datasets and data loaders
    dataset = SentenceDataset(sentences, labels)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, collate_fn=lambda batch: {
        'input_ids': pad_sequence([item['input_ids'] for item in batch], batch_first=True),
        'attention_mask': pad_sequence([item['attention_mask'] for item in batch], batch_first=True),
        'labels': torch.stack([item['label'] for item in batch])
    })
    return dataloader
