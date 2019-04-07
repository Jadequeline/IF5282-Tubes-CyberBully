from __future__ import print_function
import numpy as np
np.random.seed(1337)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.preprocessing.text import text_to_word_sequence
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.regularizers import l2
from sklearn import preprocessing
from nltk.tokenize import  TweetTokenizer
from sacremoses import  MosesDetokenizer
import gensim
import re
from bs4 import BeautifulSoup
import pandas as pd
import csv
import nltk

MAX_SEQ_LENGTH = 69
MAX_NB_WORDS = 40000
EMBEDDING_DIM = 300

slang_dict = {}
listLabel = []
listWordBully = []
textPost = []

df = pd.read_csv("data/formspring_data.csv", sep='\t')
df.head()

def to_lowercase(text):
    if(text != None and text != ""):
        text = text.lower()
    return text

def strip_html(text):
    text_new=""
    if(text != None and text != ""):
        soup = BeautifulSoup(text, "html.parser")
        text_new = soup.get_text()
    return text_new

def remove_url(text):
    return re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)

def remove_punctuation(text):
    text = re.sub(r"[^\w\d'\s]+", '', text)
    return text

def convert_unicode_039(text):
    text = re.sub(r'&#039;',"'",text)
    return text

def convert_unicode_u2019(text):
    text = re.sub(r'u2019',"'",text)
    return text

def openSlang():
    with open('data/slang_dict.txt', 'r', encoding='ISO-8859-1') as slang:
        for line in csv.reader(slang, delimiter='\t'):
            slang_dict[line[-2]] = line[-1]


def preprocess_text(text):

    text = gensim.utils.any2unicode(text)

    text = convert_unicode_039(text)

    text = convert_unicode_u2019(text)

    text = remove_url(text)

    text = strip_html(text)

    text = remove_punctuation(text)

    tknzr = TweetTokenizer()
    text = tknzr.tokenize(text)

    output = [to_lowercase(word) for word in text if not word.isdigit()]
    ww = []

    for token in output:
        if token in slang_dict.keys():
            ww.append(token.replace(token, slang_dict[token]))
        else:
            ww.append(token)

    return ww


def loadData():
    openSlang()

    listPost = df['post']
    md = MosesDetokenizer()
    for text in listPost:
        text = str(text).replace("Q:", "")
        text = str(text).replace("A:", "")
        textPost.append(md.detokenize(preprocess_text(text=str(text))))
    df['post'] = textPost

    countYes = 0
    for i in range(len(df)):

        ans1 = str(df["ans1"][i])
        ans2 = str(df["ans2"][i])
        ans3 = str(df["ans3"][i])

        bull1 = str(df["bully1"][i])
        bull2 = str(df["bully2"][i])
        bull3 = str(df["bully3"][i])

        if ans1.lower() == "yes" or ans2.lower() == "yes" or ans3.lower() == "yes":
            countYes = countYes + 1
            if countYes > 1:
                listLabel.append("yes")
                countYes = 0
            else:
                listLabel.append("no")
        else:
            listLabel.append("no")

        if bull1.lower() != "n/a" and bull1.lower() != "none" and bull1.lower() != "nan":
            listWordBully.append(bull1)
        if bull2.lower() != "n/a" and bull2.lower() != "none" and bull2.lower() != "nan":
            listWordBully.append(bull2)
        if bull3.lower() != "n/a" and bull3.lower() != "none" and bull3.lower() != "nan":
            listWordBully.append(bull3)

loadData()

print('Indexing word vectors.')

embeddings_index = {}
f = open('data/glove.6B.300d.txt',encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=False)
tokenizer.fit_on_texts(textPost)
sequences = tokenizer.texts_to_sequences(textPost)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQ_LENGTH, padding='post')
print('Shape of data tensor:', data.shape)

y = pd.get_dummies(listLabel).values

print('Preparing embedding matrix.')

nb_words = len(word_index)
embedding_matrix = np.zeros((nb_words + 1, 300))
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(nb_words + 1,
                            300,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQ_LENGTH,
                            trainable=False)
print('Embedding Layer set..')

embedding_model = Sequential()
embedding_model.add(embedding_layer)

embedding_model.compile(loss='categorical_crossentropy',
                        optimizer='rmsprop',
                        metrics=['acc']
                       )
embedding_output = embedding_model.predict(data)
print('Generated word Embeddings..')
print('Shape of Embedding_output', embedding_output.shape)


train_input = np.zeros(shape=(len(data), 69,306))
le = preprocessing.LabelEncoder()
tags = ["CC", "NN", "JJ", "VB", "RB", "IN"]
le.fit(tags)
i = 0

for sent in textPost:
    s = text_to_word_sequence(sent)
    tags_for_sent = nltk.pos_tag(s)
    sent_len = len(tags_for_sent)
    ohe = [0] * 6

    for j in range(69):
        if j < len(tags_for_sent) and tags_for_sent[j][1][:2] in tags:
            ddLe = le.transform([tags_for_sent[j][1][:2]])
            ohe[ddLe[0]] = 1
        train_input[i][j] = np.concatenate([embedding_output[i][j], ohe])
    i = i + 1
print('Concatenated Word-Embeddings and POS Tag Features...')

print('Training Model...')
model = Sequential()
model.add(Conv1D(100, 5, padding="same", input_shape=(69, 306)))
model.add(Activation("tanh"))
model.add(MaxPooling1D(pool_size=5))
model.add(Conv1D(50, 3, padding="same"))
model.add(Activation("tanh"))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("tanh"))
# softmax classifier
model.add(Dense(69, kernel_regularizer=l2(0.01)))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print('Model Trained.')

model.fit(train_input, y,
          validation_split=0.1,
          batch_size=10,
          epochs=50
         )

