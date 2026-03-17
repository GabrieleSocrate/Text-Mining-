# Text-Mining- Financial News Sentiment Analysis

## Project Overview
The goal of this project is to perform sentiment analysis on real financial news from 2018 to 2023, scraped from Yahoo Finance. Several neural network models were implemented to classify each news in one of the three plausible labels: "Positive", "Negative" or "Neutral".

The structure of the repository is as follows:

### 1. Data Loading and Cleaning
- loading the data
- ensuring the dataset is free of missing values and other inconsistencies, keeping only relevant columns

### 2. Exploratory Data Analysis and BiLSTM

### 3. Bert Models
#### DistilBert fine-tuning: 
- transformers model which was pretrained by knowledge distillation using BERT as foundational model.

#### FinBert feature extraction: 
- pre-trained model to analyze sentiment of financial text, built by further training the BERT language model in the finance domain.

#### Bert base fine-tuning:
- transformers model pretrained on generic data, considered a foundational model for sequence classification, token classification or question answering tasks.

### 4. RAG

- loading and spliting the dataset into small text chunks.

- creating embeddings and storing them in a FAISS index for efficient similarity-based retrieval of relevant context.

- implementing a question-answering system that retrieves top-k relevant chunks, builds an augmented prompt, and asks a local LLM (via Ollama) to generate context-grounded answers.

### 5. Interactive Dashboard
