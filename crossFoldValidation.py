import random
import time
from datetime import datetime
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from tensorflow import keras
from keras._tf_keras.keras.preprocessing import sequence
from keras._tf_keras.keras.preprocessing.text import Tokenizer

CSV_FILENAME = "./tweets.csv"
CSV_NO_IDS_FILENAME = "2000Tweets.csv"

MISSING = object()

# HYPERPARAMETERS
HYPERPARAMETERS = {
    "EMBEDDING_DIM": 2048,
    "NUM_HIDDEN_NODES": 2048,
    "NUM_OUTPUT_NODES": 1,
    "NUM_LAYERS": 2,
    "BIDIRECTION": True,
    "DROPOUT": 0.2,
    "NUM_TRAINING_EPOCHS": 15
}

class DatasetMapper(Dataset):
    '''
    Handles batches of dataset
    '''
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def calculate_accuracy(ground_truth, predictions):
    true_positives = 0
    true_negatives = 0

    for true, pred in zip(ground_truth, predictions):
        if (pred > 0.5) and (true == 1):
            true_positives += 1
        elif (pred < 0.5) and (true == 0):
            true_negatives += 1

    return (true_positives+true_negatives) / len(ground_truth)

class Model(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers, bidirectional, dropout):
        super(Model, self).__init__()
        self.lstm_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_features=2*self.hidden_dim, out_features=257)
        self.fc2 = nn.Linear(257, 1)

    def forward(self, x):
      device = x.device  # Get the device of the input tensor
      h = torch.zeros((2 * self.lstm_layers, x.size(0), self.hidden_dim), device=device)
      c = torch.zeros((2 * self.lstm_layers, x.size(0), self.hidden_dim), device=device)

      torch.nn.init.xavier_normal_(h)
      torch.nn.init.xavier_normal_(c)

      out = self.embedding(x)  # Ensure x is on the same device as model parameters
      out, (hidden, cell) = self.lstm(out, (h, c))
      out = self.dropout(out)
      out = torch.relu_(self.fc1(out[:, -1, :]))
      out = self.dropout(out)
      out = torch.sigmoid(self.fc2(out))

      return out


def prepare_datasets_and_model(HYPERPARAMETERS: dict):
    csv_data = pd.read_csv(CSV_NO_IDS_FILENAME)

    X = csv_data['content'].values
    Y = csv_data['class'].values

    raw_x_train, raw_x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    tokens = Tokenizer(num_words=1000)
    tokens.fit_on_texts(raw_x_train)
    sequences = tokens.texts_to_sequences(raw_x_train)
    x_train = sequence.pad_sequences(sequences, maxlen=csv_data.content.str.len().max())
    sequences = tokens.texts_to_sequences(raw_x_test)
    x_test = sequence.pad_sequences(sequences, maxlen=csv_data.content.str.len().max())

    training_set = DatasetMapper(x_train, y_train)
    test_set = DatasetMapper(x_test, y_test)

    BATCH_SIZE = 64
    train_loader = DataLoader(training_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(
        HYPERPARAMETERS["EMBEDDING_DIM"],
        HYPERPARAMETERS["NUM_HIDDEN_NODES"],
        HYPERPARAMETERS["NUM_LAYERS"],
        HYPERPARAMETERS["BIDIRECTION"],
        HYPERPARAMETERS["DROPOUT"]
    )

    model = model.to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=1e-4)
    return model, optimizer, train_loader, test_loader, y_train, y_test

def train(model, loader, optimizer, device) -> list:
    epoch_loss = 0.0
    epoch_acc = 0.0

    model.train()

    epoch_predictions = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device).long()  # Move to GPU/CPU
        y_batch = y_batch.to(device).float()  # Move to GPU/CPU

        optimizer.zero_grad()

        predictions = model(x_batch)
        predictions = torch.squeeze(predictions)

        loss = F.binary_cross_entropy(predictions, y_batch)
        loss.backward()

        optimizer.step()

        epoch_predictions += list(predictions.squeeze().detach().cpu().numpy())

    return epoch_predictions, loss

def evaluate(model, loader, device) -> list:
    model.eval()

    epoch_predictions = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device).long()

            y_pred = model(x_batch)
            epoch_predictions += list(y_pred.detach().cpu().numpy())
    return epoch_predictions

def run(auto_save: bool):
    model, optimizer, train_loader, test_loader, y_train, y_test = prepare_datasets_and_model(HYPERPARAMETERS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_acc = 0

    for epoch in range(1, HYPERPARAMETERS["NUM_TRAINING_EPOCHS"] + 1):
        train_predictions, loss = train(model, train_loader, optimizer, device)
        test_predictions = evaluate(model, test_loader, device)

        train_accuracy = calculate_accuracy(y_train, train_predictions)
        test_accuracy = calculate_accuracy(y_test, test_predictions)

        final_acc = test_accuracy
        print(f"Epoch: {epoch+1}, loss: {loss.item():.5f}, Train accuracy: {train_accuracy:.5f}, Test accuracy: {test_accuracy:.5f}")

if __name__ == "__main__":
    run(auto_save=True)
