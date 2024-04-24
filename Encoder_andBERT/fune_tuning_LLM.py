import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from load_data import bert_dataloader

# Load pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained("bert-base-multilingual-cased")

# Extend BERT for classification
class BertForClassification(nn.Module):
    def __init__(self, num_classes):
        super(BertForClassification, self).__init__()
        self.bert = model
        self.classifier = nn.Linear(model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)


batcisize = 4
num_epochs = 30
device = "cuda" if torch.cuda.is_available() else "cpu"

for dataset in ["german", "socc"]:
    if dataset == "german":
        data = pd.read_csv("./data/german hatespeech refugees.csv")
        sentences = data["Tweet"].tolist()
        labels = list(np.load("./data/german_label_expert1.npy"))
    elif dataset == "socc":
        data = pd.read_csv("./data/SFU_constructiveness_toxicity_corpus.csv")
        sentences = data["comment_text"].tolist()
        labels = list(np.load("./data/socc_label.npy"))

    train_sentences, test_sentences, train_labels, test_labels = train_test_split(sentences, labels, train_size=0.8, random_state=42)
    train_sentences, val_sentences, train_labels, val_labels = train_test_split(train_sentences, train_labels, train_size=0.75, random_state=42)

    train_dataloader = bert_dataloader(train_sentences, train_labels, batcisize)
    val_dataloader = bert_dataloader(val_sentences, val_labels, batcisize)
    test_dataloader = bert_dataloader(test_sentences, test_labels, batcisize)

    num_classes = 2  # Number of classes for classification

    model_with_classifier = BertForClassification(num_classes)

    # Training loop
    optimizer = torch.optim.Adam(model_with_classifier.parameters(), lr=1e-5)
    loss_fn = nn.CrossEntropyLoss()



    best_val_f1 = 0.0

    for epoch in range(num_epochs):
        # Train
        model_with_classifier.to(device)
        model_with_classifier.train()
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model_with_classifier(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model_with_classifier.eval()
        val_labels = []
        val_predictions = []

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits = model_with_classifier(input_ids, attention_mask)
                predictions = torch.argmax(logits, dim=1)

                val_labels.extend(labels.cpu().numpy())
                val_predictions.extend(predictions.cpu().numpy())

        val_f1 = f1_score(val_labels, val_predictions, average='weighted')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model = model_with_classifier.state_dict()

            # Test on test dataloader
            test_labels = []
            test_predictions = []

            with torch.no_grad():
                for batch in test_dataloader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    logits = model_with_classifier(input_ids, attention_mask)
                    predictions = torch.argmax(logits, dim=1)

                    test_labels.extend(labels.cpu())
                    test_predictions.extend(predictions.cpu())

            test_labels, test_predictions = torch.tensor(test_labels), torch.tensor(test_predictions)
            test_accuracy = (test_predictions == test_labels).float().mean()
            test_precision = (test_predictions[test_labels == 1] == 1).float().sum() / (test_predictions == 1).float().sum().clamp(min=1e-8)
            test_recall = (test_predictions[test_labels == 1] == 1).float().sum() / (test_labels == 1).float().sum().clamp(min=1e-8)
            test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall).clamp(min=1e-8)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss.item():.4f}, Validation F1 Score: {val_f1:.4f}")
        # if val_f1 == best_val_f1:
        #     print(f"Best Validation F1 Score: {best_val_f1:.4f}")
        #     print(f"Test Accuracy: {test_accuracy:.4f}")
        #     print(f"Test Precision: {test_precision:.4f}")
        #     print(f"Test Recall: {test_recall:.4f}")
        #     print(f"Test F1 Score: {test_f1:.4f}")

    print(f"The test results on {dataset} in the loop with the best val results is "
          f"accuracy: {test_accuracy:.4f}, precision: {test_precision:.4f}, recall: {test_recall:.4f}, f1 score: {test_f1:.4f}")

