# Sentiment Analysis using RNN and Word Embeddings

Sentiment analysis on Amazon reviews using deep learning models (SimpleRNN and LSTM). Preprocess the dataset, split it into training and validation sets, and build a vocabulary for word embedding. The project features real-time prediction for user-inputted reviews and includes a report detailing model summaries and optimal hyperparameters. Perfect for NLP enthusiasts exploring text classification techniques.

## Introduction

Sentiment analysis is a common task in natural language processing (NLP) that aims to determine the sentiment or opinion expressed in a piece of text. This project leverages deep learning techniques, specifically RNN and LSTM models, to classify the sentiment of Amazon reviews into positive, neutral, or negative categories.

## Features

- **Data Preprocessing:** Includes cleaning and preprocessing text data.
- **Tokenization and Padding:** Converts text data into a numerical format suitable for model training.
- **Model Creation:** Defines and initializes RNN and LSTM models.
- **Training and Validation:** Implements training loops with performance tracking.

- ## Dataset

The project uses the Amazon reviews dataset, which contains customer reviews and ratings. The dataset includes the following columns:
- `sentiments`: The sentiment of the review (positive, neutral, negative).
- `cleaned_review`: The preprocessed review text.
- `cleaned_review_length`: The length of the cleaned review.
- `review_score`: The review score given by the customer.

- ## Model Training

The project defines and trains two types of models:
- **RNN Model:** A simple Recurrent Neural Network for sentiment classification.

    **RNN Model Performance:**


    RNN Model Accuracy: 0.8480392098426819


    RNN Model rnn_loss: 0.47567227482795715


- **LSTM Model:** A Long Short-Term Memory network, which is more effective for capturing long-term dependencies in text data.
 
   **LSTM Model Performance:**
  
    lstm Model Accuracy: 0.8777393102645874
  
    lstm Model rnn_loss: 0.411745548248291

Both models are trained using the Adam optimizer and cross-entropy loss function.
