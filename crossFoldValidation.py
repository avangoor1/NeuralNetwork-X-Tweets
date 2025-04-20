import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing import sequence

# Hyperparameters
HYPERPARAMETERS = {
    "EMBEDDING_DIM": 2048,
    "NUM_HIDDEN_NODES": 2048,
    "NUM_OUTPUT_NODES": 1,
    "NUM_LAYERS": 2,
    "BIDIRECTION": True,
    "DROPOUT": 0.2,
    "NUM_TRAINING_EPOCHS": 10,
    "BATCH_SIZE": 64,
    "NUM_FOLDS": 5  # Adjust based on the number of folds you have
}

class DatasetMapper(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

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
        self.fc1 = nn.Linear(2 * hidden_dim, 257)
        self.fc2 = nn.Linear(257, 1)

    def forward(self, x):
        h = torch.zeros((2 * self.lstm_layers, x.size(0), self.hidden_dim))
        c = torch.zeros((2 * self.lstm_layers, x.size(0), self.hidden_dim))
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
    train_file = f"foldTrain{fold}.csv"
    test_file = f"foldTest{fold}.csv"
    
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    
    X_train, Y_train = train_data['content'].values, train_data['class'].values
    X_test, Y_test = test_data['content'].values, test_data['class'].values
    
    tokenizer = Tokenizer(num_words=1000)
    tokenizer.fit_on_texts(X_train)
    
    x_train = sequence.pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=train_data.content.str.len().max())
    x_test = sequence.pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=test_data.content.str.len().max())
    
    return DatasetMapper(x_train, Y_train), DatasetMapper(x_test, Y_test)

def train(model, loader, optimizer):
    model.train()
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        x = x_batch.type(torch.LongTensor)
        y = y_batch.type(torch.FloatTensor)
        predictions = model(x).squeeze()
        loss = F.binary_cross_entropy(predictions, y)
        loss.backward()
        optimizer.step()
    return loss.item()

def evaluate(model, loader, dataset):
    model.eval()
    predictions = []
    ground_truths = []
    tweet_texts = dataset.x  # Store original tweet texts
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x = x_batch.type(torch.LongTensor)
            y_pred = model(x).squeeze().detach().numpy()
            predictions.extend(y_pred)
            ground_truths.extend(y_batch.numpy())

    return tweet_texts, predictions, ground_truths

def calculate_accuracy(ground_truth, predictions):
    correct = sum((p > 0.5) == (t == 1) for p, t in zip(predictions, ground_truth))
    return correct / len(ground_truth)

def run_cross_validation():
    total_accuracy = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for fold in range(1, HYPERPARAMETERS["NUM_FOLDS"] + 1):
        print(f"Processing Fold {fold}...")
        train_set, test_set = load_data(fold)

        train_loader = DataLoader(train_set, batch_size=HYPERPARAMETERS["BATCH_SIZE"], shuffle=True)
        test_loader = DataLoader(test_set)

        model = Model(
            HYPERPARAMETERS["EMBEDDING_DIM"],
            HYPERPARAMETERS["NUM_HIDDEN_NODES"],
            HYPERPARAMETERS["NUM_LAYERS"],
            HYPERPARAMETERS["BIDIRECTION"],
            HYPERPARAMETERS["DROPOUT"]
        ).to(device)
        optimizer = optim.RMSprop(model.parameters(), lr=1e-4)

        for epoch in range(1, HYPERPARAMETERS["NUM_TRAINING_EPOCHS"] + 1):
            loss = train(model, train_loader, optimizer)
            print(f"Fold {fold}, Epoch {epoch}: Loss = {loss:.5f}")

        # Evaluate and save predictions
        tweet_texts, test_predictions, ground_truths = evaluate(model, test_loader, test_set)
        accuracy = calculate_accuracy(ground_truths, test_predictions)
        print(f"Fold {fold} Accuracy: {accuracy:.5f}")
        total_accuracy += accuracy

        # Save predictions and ground truth to a CSV file
        output_df = pd.DataFrame({
            "Tweet_Text": tweet_texts,
            "Predicted_Score": test_predictions,
            "Ground_Truth": ground_truths
        })
        output_df.to_csv(f"fold_predictions_{fold}.csv", index=False)
        print(f"Saved predictions for Fold {fold} to fold_predictions_{fold}.csv")

    avg_accuracy = total_accuracy / HYPERPARAMETERS["NUM_FOLDS"]
    print(f"Average Accuracy across {HYPERPARAMETERS['NUM_FOLDS']} folds: {avg_accuracy:.5f}")


if __name__ == "__main__":
    run_cross_validation()
