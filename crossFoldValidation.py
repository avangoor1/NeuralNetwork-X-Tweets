# File contents: Cross fold validation to create your neural network model, add the training data set file 
# yourself. And you can write where you want that file to be stored. If the program is taking too long to run
# on your local computer, try running it on Google Colab.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing import sequence
import numpy as np
from sklearn.model_selection import train_test_split

# Hyperparameters
HYPERPARAMETERS = {
    "EMBEDDING_DIM": 2048,
    "NUM_HIDDEN_NODES": 2048,
    "NUM_OUTPUT_NODES": 1,
    "NUM_LAYERS": 2,
    "BIDIRECTION": True,
    "DROPOUT": 0.2,
    "NUM_TRAINING_EPOCHS": 16,
    "BATCH_SIZE": 64
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clear_cuda_memory():
    torch.cuda.empty_cache()

class DatasetMapper(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.long, device=device)
        self.y = torch.tensor(y, dtype=torch.float, device=device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class Model(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_layers, bidirectional, dropout):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(embedding_dim, hidden_dim).to(device)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True
        ).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.fc1 = nn.Linear(2 * hidden_dim, 257).to(device)
        self.fc2 = nn.Linear(257, 1).to(device)

    def forward(self, x):
        h = torch.zeros((2 * self.lstm.num_layers, x.size(0), self.lstm.hidden_size), device=device)
        c = torch.zeros((2 * self.lstm.num_layers, x.size(0), self.lstm.hidden_size), device=device)
        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)
        out = self.embedding(x)
        out, (hidden, cell) = self.lstm(out, (h, c))
        out = self.dropout(out)
        out = torch.relu(self.fc1(out[:, -1, :]))
        out = self.dropout(out)
        out = torch.sigmoid(self.fc2(out))
        return out

def load_data():
    train_file = "" # add the path to the file you want to train from, checkout the folder "Cross Fold Validation Sets" for an example
    train_data = pd.read_csv(train_file)

    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(train_data['content'].values)

    x_data = sequence.pad_sequences(tokenizer.texts_to_sequences(train_data['content'].values))
    y_data = train_data['class'].values

    # Split data into training (80%) and testing (20%)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    return DatasetMapper(x_train, y_train), DatasetMapper(x_test, y_test)

def train(model, loader, optimizer):
    model.train()
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        x, y = x_batch.to(device), y_batch.to(device)
        predictions = model(x).squeeze()
        loss = F.binary_cross_entropy(predictions, y)
        loss.backward()
        optimizer.step()
    return loss.item()

def run_training():
    clear_cuda_memory()
    train_set, test_set = load_data()
    train_loader = DataLoader(train_set, batch_size=HYPERPARAMETERS["BATCH_SIZE"], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=HYPERPARAMETERS["BATCH_SIZE"], shuffle=False)

    model = Model(HYPERPARAMETERS["EMBEDDING_DIM"], HYPERPARAMETERS["NUM_HIDDEN_NODES"],
                  HYPERPARAMETERS["NUM_LAYERS"], HYPERPARAMETERS["BIDIRECTION"],
                  HYPERPARAMETERS["DROPOUT"]).to(device)

    optimizer = optim.RMSprop(model.parameters(), lr=1e-4)

    for epoch in range(1, HYPERPARAMETERS["NUM_TRAINING_EPOCHS"] + 1):
        loss = train(model, train_loader, optimizer)
        print(f"Epoch {epoch}: Loss = {loss:.5f}")

    # Save trained model
    model_path = "" # add the path you want to save the model to
    torch.save(model.state_dict(), model_path)
    print(f"Saved trained model to {model_path}")

if __name__ == "__main__":
    run_training()
