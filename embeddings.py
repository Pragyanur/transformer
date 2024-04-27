import pandas as pd
import numpy as np
import tensorflow as tf 
import nltk
import keras
from keras.preprocessing.sequence import pad_sequences

TF_ENABLE_ONEDNN_OPTS=0

dataFrame = pd.read_csv('brown.csv')
sentences = dataFrame['tokenized_text']

documents = []
vocabulary = []
train_data = []

for s in sentences:
    words = list(w for w in nltk.word_tokenize(str(s)))
    documents.append(words)
    vocabulary.extend(words)

vocabulary = set(vocabulary)
VOCABULARY_SIZE = len(vocabulary)

word2idx = {word: idx for idx, word in enumerate(vocabulary)}
idx2word = dict(enumerate(vocabulary))

# print(word2idx[str(idx2word[0])], idx2word[0])

# Generate training data
WINDOW_SIZE = 10
for sentence in documents:
    for i, target_word in enumerate(sentence):
        context_words = [sentence[j] for j in range(max(0, i - WINDOW_SIZE), min(i + WINDOW_SIZE + 1,len(sentence) )) if j>=0 and j != i]
        train_data.append((context_words, target_word))

# print(train_data[0:3])

# Generate input and output pairs for CBOW
train_inputs, train_labels = [], []
for context_words, target_word in train_data:
    context_idxs = [word2idx[word] for word in context_words]
    train_inputs.append(context_idxs)
    train_labels.append(word2idx[target_word])

# print(train_inputs, train_labels)

# Convert to numpy arrays
train_inputs = pad_sequences(train_inputs, maxlen=WINDOW_SIZE*2)
train_labels = np.array(train_labels)

# Define CBOW model
EMBEDDING_DIM = 10
cbow_model = keras.Sequential([
    keras.layers.Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBEDDING_DIM),
    keras.layers.Flatten(),
    keras.layers.Dense(units=VOCABULARY_SIZE, activation='softmax')
])

# Compile and train the CBOW model
cbow_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
cbow_model.fit(train_inputs, train_labels, epochs=10, verbose=1)

# Get the learned word embeddings
word_embeddings = cbow_model.get_weights()[0]

# Print the word embeddings
for i, embedding in enumerate(word_embeddings):
    word = idx2word[i]
    print(f"Word: {word}, Embedding: {embedding}")


