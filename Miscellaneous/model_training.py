# Another way to train your model if not using cross fold validation

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
CSV_NO_IDS_FILENAME = "./tweets_no_ids.csv"

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
        else:
            pass
            
    return (true_positives+true_negatives) / len(ground_truth)

class Model(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,n_layers,bidirectional,dropout):
        super(Model, self).__init__()
        self.lstm_layers = n_layers
        self.hidden_dim = hidden_dim
        # self.embedding = nn.Embedding(embedding_dim, hidden_dim, padding_idx=0)
        self.embedding = nn.Embedding(embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size = embedding_dim,
            hidden_size = hidden_dim,
            num_layers = n_layers,
            bidirectional = bidirectional,
            batch_first = True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_features=2*self.hidden_dim, out_features=257)
        self.fc2 = nn.Linear(257, 1)

    def forward(self,x):
        h = torch.zeros((2 * self.lstm_layers, x.size(0), self.hidden_dim))
        c = torch.zeros((2 * self.lstm_layers, x.size(0), self.hidden_dim))
        
        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)


        out = self.embedding(x)
        out, (hidden, cell) = self.lstm(out, (h,c))
        out = self.dropout(out)
        out = torch.relu_(self.fc1(out[:,-1,:]))
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

def train(model, loader, optimizer) -> list:
    
    epoch_loss = 0.0
    epoch_acc = 0.0
    
    model.train()
    
    epoch_predictions = []

    for x_batch, y_batch in loader:
        
        # Zero out optimizer between batches
        optimizer.zero_grad()
        
        x = x_batch.type(torch.LongTensor)
        y = y_batch.type(torch.FloatTensor)
        
        # Forward pass
        predictions = model(x)
        
        predictions = torch.squeeze(predictions)

        # Backpropogation
        loss = F.binary_cross_entropy(predictions, y)
        loss.backward()
        
        # Update optimizer
        optimizer.step()

        epoch_predictions += list(predictions.squeeze().detach().numpy())

    return epoch_predictions, loss

def evaluate(model, loader) -> list:

    # Switch model from training mode to evaluation mode, prevents dropout
    model.eval()
    
    epoch_predictions = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x = x_batch.type(torch.LongTensor)
            y = y_batch.type(torch.FloatTensor)
            
            y_pred = model(x)
            epoch_predictions += list(y_pred.detach().numpy())
    return epoch_predictions
    


def run(auto_save: bool):
    model, optimizer, train_loader, test_loader, y_train, y_test = prepare_datasets_and_model(HYPERPARAMETERS)
    final_acc = 0
    for epoch in range(1, HYPERPARAMETERS["NUM_TRAINING_EPOCHS"] + 1):
        
        train_predictions, loss = train(model, train_loader, optimizer)
        
        test_predictions = evaluate(model, test_loader)
        
        train_accuracy = calculate_accuracy(y_train, train_predictions)
        test_accuracy = calculate_accuracy(y_test, test_predictions)

        final_acc = test_accuracy
        print("Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f" % (epoch+1, loss.item(), train_accuracy, test_accuracy))

    print(f"Final model accuracy: {final_acc:.3f}.  Save this model? (Y/n)")
    if not auto_save:
         do_save: str = input(">> ")
    if auto_save or do_save.casefold().startswith("y"):
        current_date: str = datetime.today().strftime('%Y-%m-%d') # YYYY-MM-DD
        # model_state_filename: str = f"TrainedModel_{current_date}_{final_acc:.3f}_{HYPERPARAMETERS['VOCAB_SIZE']}.pt"
        model_state_filename: str = f"TrainedModel_{current_date}_{final_acc:.3f}.pt"
        torch.save(model.state_dict(), model_state_filename)

if __name__ == "__main__":
    run(auto_save=True)