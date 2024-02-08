# LSTM Time Series Prediction
This project focuses on utilizing LSTM models for predicting time series data, specifically the closing prices of Amazon stocks.

## Dependencies
numpy
pandas
matplotlib
torch
sklearn
## Data Preparation
Data is sourced from a CSV file named 'data-amz'. The 'Date' and 'Close' columns are isolated, with the 'Date' column converted to datetime format. Visualization of the closing prices over time is facilitated through plotting.

A function, prepare_dataframe_for_lstm, is employed to ready the data for LSTM modeling. This function shifts the 'Close' column by a specified number of steps and integrates these shifted columns into the dataframe.

The data undergoes scaling via sklearn's MinMaxScaler, normalizing it to a range of -1 to 1.

Data segmentation allocates 95% for training and 5% for testing.

## Model Architecture
The model architecture comprises an LSTM layer and a fully connected layer, implemented using PyTorch. The LSTM layer processes input sequences, producing hidden states for each element. The final hidden state feeds into the fully connected layer, yielding a single value, representing the prediction for the subsequent time step.

## Training
The train_one_epoch function conducts model training for a single epoch. It accepts training data and the model as inputs, printing batch-wise loss every 100 batches.

## Validation
Validation of the model on testing data is executed via the validate_one_epoch function. This function computes the loss for the testing dataset.

## Implementation
Execute the provided Python script to utilize this project. The script handles data loading, LSTM model preparation, training, and subsequent validation on testing data.
