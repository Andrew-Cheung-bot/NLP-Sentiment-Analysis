import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

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

