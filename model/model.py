# Import library
import warnings
warnings.filterwarnings("ignore")
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

nltk.download('punkt')
nltk.download('wordnet')

# Inisialisasi lemmatizer
lemmatizer = WordNetLemmatizer()

# Memuat data training
words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open("model/data.json").read()
intents = json.loads(data_file)

# Preprocessing data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenisasi kata
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        
        # Tambahkan ke kelas
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatisasi dan normalisasi
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# sort classes
classes = sorted(list(set(classes)))

# documents = combination between patterns and intents
print (len(documents), "documents")

# classes = intents
print (len(classes), "classes", classes)

# words = all words, vocabulary
print (len(words), "unique lemmatized words", words)
# Simpan vocabulary dan labels
pickle.dump(words, open('model/texts.pkl', 'wb'))
pickle.dump(classes, open('model/labels.pkl', 'wb'))

# Membuat data training
training = []

# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
   
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# shuffle features and converting it into numpy arrays
random.shuffle(training)

# Convert training data properly
train_x = np.array([np.array(i[0]) for i in training], dtype=np.float32)
train_y = np.array([np.array(i[1]) for i in training], dtype=np.float32)

print("Training data created")

# Create NN model to predict the responses
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fitting and saving the model 
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('model/models.h5', hist) # we will pickle this model to use in the future
print("\n")
print("*"*50)
print("\nModel Created Successfully!")

# Evaluating the model
loss, accuracy = model.evaluate(np.array(train_x), np.array(train_y))
print('Accuracy:', accuracy)
print('Loss:',loss)