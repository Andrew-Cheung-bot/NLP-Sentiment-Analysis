from models.transformer import TransformerClassifier
from load_data import load_splited_data
import torch
import torch.nn as nn
import torch.optim as optim
from train_test_func import train
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import os
import pandas as pd


(
    X_train_german_sen, X_test_german_sen,
    X_train_german_word, X_test_german_word,
    y_train_german1, y_test_german1,
    y_train_german2, y_test_german2,
    X_train_socc_sen, X_test_socc_sen,
    X_train_socc_word, X_test_socc_word,
    y_train_socc, y_test_socc
    ) = load_splited_data(training_size=0.8)

device = "cuda" if torch.cuda.is_available() else "cpu"

for dataset_name in ["german", "socc"]:
    if dataset_name == "german":
        batch_size = 32
        num_epochs = 300
        num_heads_list = [1, 2, 4]
        hidden_dim_list = [4, 8, 16, 32]
        num_layers_list = [2, 4, 6]
        dropout_list = [0.1, 0.2, 0.3]

        X_train_german_word, X_val_german_word, y_train_german1, y_val_german1 = train_test_split(X_train_german_word, y_train_german1, train_size=0.75, random_state=42)
        train_dataset = TensorDataset(torch.from_numpy(X_train_german_word), torch.from_numpy(y_train_german1))
        val_dataset = TensorDataset(torch.from_numpy(X_val_german_word), torch.from_numpy(y_val_german1))
        test_dataset = TensorDataset(torch.from_numpy(X_test_german_word), torch.from_numpy(y_test_german1))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    elif dataset_name == "socc":
        batch_size = 64
        num_epochs = 500
        num_heads_list = [1, 2, 4]
        hidden_dim_list = [4, 8, 16, 32]
        num_layers_list = [2, 4, 6]
        dropout_list = [0.1, 0.3]

        X_train_socc_word, X_val_socc_word, y_train_socc, y_val_socc = train_test_split(X_train_socc_word,
                                                                                                  y_train_socc,
                                                                                                  train_size=0.75,
                                                                                                  random_state=42)
        train_dataset = TensorDataset(torch.from_numpy(X_train_socc_word), torch.from_numpy(y_train_socc))
        val_dataset = TensorDataset(torch.from_numpy(X_val_socc_word), torch.from_numpy(y_val_socc))
        test_dataset = TensorDataset(torch.from_numpy(X_test_socc_word), torch.from_numpy(y_test_socc))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



    os.makedirs("./results", exist_ok=True)
    txt_path = f"./results/results_transformer_{dataset_name}.txt"
    if not os.path.exists(os.path.dirname(txt_path)):
        os.makedirs(os.path.dirname(txt_path))
    def append_results_to_file(path, line):
        with open(path, "a") as file:
            file.write(line)

    with open(txt_path, "w") as file:
        headers = "num_heads,hidden_dim,num_layers,dropout,test_accuracy,test_precision,test_recall,test_f1score\n"
        file.write(headers)



    for num_heads in num_heads_list:
        for hidden_dim in hidden_dim_list:
            for num_layers in num_layers_list:
                for dropout in dropout_list:
                    model = TransformerClassifier(embedding_dim=300, num_heads=num_heads,
                                                  hidden_dim=hidden_dim, num_layers=num_layers,
                                                  num_classes=2, dropout=dropout)
                    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
                    test_acc, test_precision, test_recall, test_f1score = train(model, train_dataloader, val_dataloader, test_dataloader, optimizer=optimizer, num_epochs=num_epochs, device=device)
                    print("Test accuracy:", test_acc)
                    print("Test precision:", test_precision)
                    print("Test recall:", test_recall)
                    print("Test f1score:", test_f1score)

                    result_line = f"{num_heads},{hidden_dim},{num_layers},{dropout},{test_acc},{test_precision},{test_recall},{test_f1score}\n"
                    append_results_to_file(txt_path, result_line)


    results_transformer = pd.read_csv(f'./results/results_transformer_{dataset_name}.txt')
    print(f"The best results of transformer on {dataset_name} dataset is {results_transformer.loc[results_transformer['test_f1score'].idxmax()]}.")

