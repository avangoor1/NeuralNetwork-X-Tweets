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


# Hyperparameters
HYPERPARAMETERS = {
    "EMBEDDING_DIM": 2048,
    "NUM_HIDDEN_NODES": 2048,
    "NUM_OUTPUT_NODES": 1,
    "NUM_LAYERS": 2,
    "BIDIRECTION": True,
    "DROPOUT": 0.2,
    "NUM_TRAINING_EPOCHS": 16,
    "BATCH_SIZE": 64,
    "NUM_FOLDS": 5
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

def load_data(fold):
    train_file = f"/content/foldTrain{fold}.csv"
    test_file = f"/content/foldTest{fold}.csv"
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(train_data['content'].values)
    x_train = sequence.pad_sequences(tokenizer.texts_to_sequences(train_data['content'].values))
    x_test = sequence.pad_sequences(tokenizer.texts_to_sequences(test_data['content'].values))
    return DatasetMapper(x_train, train_data['class'].values), DatasetMapper(x_test, test_data['class'].values)

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

def evaluate(model, loader, dataset):
    model.eval()
    predictions, ground_truths, tweet_texts = [], [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x, y = x_batch.to(device), y_batch.to(device)
            y_pred = model(x).squeeze().cpu().numpy()
            predictions.extend(y_pred.tolist() if y_pred.ndim > 0 else [y_pred.item()])
            ground_truths.extend(y.cpu().numpy())
            tweet_texts.extend(dataset.x.cpu().numpy())

    # print(f"Dataset X: {dataset.x.cpu().numpy().shape}")
    print(f"Predictions: {len(predictions)}")
    print(f"Ground Truths: {len(ground_truths)}")
    return tweet_texts, np.array(predictions).ravel(), np.array(ground_truths).ravel()


def calculate_accuracy(ground_truth, predictions):
    correct = sum((p > 0.5) == (t == 1) for p, t in zip(predictions, ground_truth))
    return correct / len(ground_truth)

def run_cross_validation():
    total_accuracy = 0
    for fold in range(1, HYPERPARAMETERS["NUM_FOLDS"] + 1):
        clear_cuda_memory()
        print(f"Processing Fold {fold}...")
        train_set, test_set = load_data(fold)
        train_loader = DataLoader(train_set, batch_size=HYPERPARAMETERS["BATCH_SIZE"], shuffle=True)
        test_loader = DataLoader(test_set)
        model = Model(HYPERPARAMETERS["EMBEDDING_DIM"], HYPERPARAMETERS["NUM_HIDDEN_NODES"],
                      HYPERPARAMETERS["NUM_LAYERS"], HYPERPARAMETERS["BIDIRECTION"],
                      HYPERPARAMETERS["DROPOUT"]).to(device)
        optimizer = optim.RMSprop(model.parameters(), lr=1e-4)
        for epoch in range(1, HYPERPARAMETERS["NUM_TRAINING_EPOCHS"] + 1):
            loss = train(model, train_loader, optimizer)
            print(f"Fold {fold}, Epoch {epoch}: Loss = {loss:.5f}")
        tweet_texts, test_predictions, ground_truths = evaluate(model, test_loader, test_set)
        accuracy = calculate_accuracy(ground_truths, test_predictions)
        print(f"Fold {fold} Accuracy: {accuracy:.5f}")
        total_accuracy += accuracy

        min_length = min(len(tweet_texts), len(test_predictions), len(ground_truths))

        # Save predictions and ground truth to a CSV file
        output_df = pd.DataFrame({
            "Tweet_Text": tweet_texts[:min_length],  # Trim to min length
            "Predicted_Score": test_predictions[:min_length],
            "Ground_Truth": ground_truths[:min_length]
        })
        output_df.to_csv(f"/content/fold_predictions_{fold}.csv", index=False)
        print(f"Saved predictions for Fold {fold} to fold_predictions_{fold}.csv")

        # Save trained model for this fold
        model_path = f"/content/model_fold_{fold}.pt"
        torch.save(model.state_dict(), model_path)
        print(f"Saved model for Fold {fold} to {model_path}")

    avg_accuracy = total_accuracy / HYPERPARAMETERS["NUM_FOLDS"]
    print(f"Average Accuracy across {HYPERPARAMETERS['NUM_FOLDS']} folds: {avg_accuracy:.5f}")

if __name__ == "__main__":
    run_cross_validation()
