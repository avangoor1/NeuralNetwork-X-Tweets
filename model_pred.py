# This program will use the model that you created to test against a set of unseen tweets. This can use used
# when you have a model created and want to test it against unseen tweets to understand the results of a specific
# group you point the tool at.

import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader
import csv

from keras._tf_keras.keras.preprocessing import sequence
from keras._tf_keras.keras.preprocessing.text import Tokenizer

from Miscellaneous.model_training import DatasetMapper, Model, HYPERPARAMETERS

CSV_NO_IDS_FILENAME = "Original Tweets Data/tweets_no_ids.csv"
COUNT = 0

def load_saved_model(filename: str) -> Model:
    components = filename.split("_")
    VOCAB_SIZE = components[-1]
    model = Model(
        HYPERPARAMETERS["EMBEDDING_DIM"],
        HYPERPARAMETERS["NUM_HIDDEN_NODES"],
        HYPERPARAMETERS["NUM_LAYERS"],
        HYPERPARAMETERS["BIDIRECTION"],
        HYPERPARAMETERS["DROPOUT"]
    )
    model.load_state_dict(torch.load(filename))
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # Make sure to set model to evaluation mode, not training
    model.eval()
    return model

def load_tokenizer() -> Tokenizer:
    csv_data = pd.read_csv(CSV_NO_IDS_FILENAME)

    X = csv_data['content'].values
    Y = csv_data['class'].values

    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X)
    return tokenizer

def predict(model: Model, input_filename: str):
    # Read in data in same format and tokenization as training data was.
    csv_data = pd.read_csv(input_filename, header=None)

    global COUNT

    # Print each string from the CSV file
    # print("Input strings:")
    for index, text in enumerate(csv_data[0].values):
        structure[COUNT] = [text]
        COUNT += 1
        # print(f"Row {index + 1}: {text}")

    num_inputs = csv_data[0].size
    dummy_y = [0 for _ in range(num_inputs)]

    tokenizer = load_tokenizer()

    input_sequences = tokenizer.texts_to_sequences(csv_data[0].values)
    padded_input = sequence.pad_sequences(input_sequences, maxlen=csv_data[0].str.len().max())

    inference_set = DatasetMapper(padded_input, dummy_y)

    BATCH_SIZE = 64
    inference_loader = DataLoader(inference_set, batch_size=BATCH_SIZE)

    # input_data = torch.tensor(padded_input)
    # input_data = torch.squeeze(input_data, 0)

    # print(input_data, input_data.size())

    model.eval()
    prediction_accumulator = []
    for inference_input, _ in inference_loader:
        with torch.no_grad():
            predictions = model(inference_input)
            prediction_accumulator.append(predictions)
    return prediction_accumulator

if __name__ == "__main__":
    
    # Open a file to write our predictions
    outputFile = open("Tweet_Data.json", "w")
    output = open("output.csv", "w")

    #dictionary to output the data
    structure = {}

    MODEL_FILENAME = "TrainedModels/trained_model_1600.pt" # Choose from saved models in the Trained Models folder
    INPUT_FILENAME = "" # One Column, Header of "content" format for the test set you want to test your neural network on
    model: Model = load_saved_model(MODEL_FILENAME)
    all_predictions = None 


    predictions = predict(model, INPUT_FILENAME)

    COUNT = 0

    # print(predictions)

    all_predictions = torch.cat(predictions, dim=0)
    print(all_predictions)
    # Convert the combined tensor into a list of values
    values = all_predictions.squeeze().tolist()  # Use squeeze() to remove extra dimensions if necessary

    # tensor_data = predictions[0]

    # values = list(tensor_data.squeeze().tolist())  # Use squeeze() to remove dimensions of size 1
    print("---------------------------------------------")
    print(values)
    print("---------------------------------------------")

    # Open the filteredTweets.csv file in read mode
    with open("filteredTweets.csv", mode='r', newline='', encoding='utf-8') as input_file:
        csv_reader = list(csv.reader(input_file))  

    # Calculate average of `values`
    if values:  # Check if values is not empty
        average = sum(values) / len(values)
        print("Temperature Average: ", average)

    # Open the filteredTweets.csv file in write mode to add `values` to each row
    with open("filteredTweets.csv", mode='w', newline='', encoding='utf-8') as output_file:
        csv_writer = csv.writer(output_file)
        
        # Loop through each row in csv_reader
        for i, row in enumerate(csv_reader):
            row.append(str(values[i]))  
            csv_writer.writerow(row)  
