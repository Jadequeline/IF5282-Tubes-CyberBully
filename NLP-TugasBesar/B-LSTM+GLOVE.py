
import pandas as pd
import os
import re
from bs4 import BeautifulSoup
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import *
from keras.initializers import Constant
from nltk.tokenize import  TweetTokenizer
import gensim
import csv
from sacremoses import  MosesDetokenizer
import pickle


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

wordBully = []
for text in listWordBully:
    wordBully.append(preprocess_text(text=str(text)))

max_features = 20000

tokenizer = Tokenizer(num_words=max_features, split=' ', oov_token='<unw>', filters=' ')
tokenizer.fit_on_texts(df['post'].values)

X = tokenizer.texts_to_sequences(df['post'].values)

X = pad_sequences(X, 50)

y = pd.get_dummies(listLabel).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

print("test set size " + str(len(X_test)))

embeddings_index = {}
f = open('data/glove.6B.300d.txt',encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

num_words = min(max_features, len(word_index)) + 1
print(num_words)

embedding_dim = 300

embedding_matrix = np.zeros((num_words, embedding_dim))

for word, i in word_index.items():
    if i > max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix[i] = np.random.randn(embedding_dim)

model = Sequential()
model.add(Embedding(num_words,
                    embedding_dim,
                    embeddings_initializer=Constant(embedding_matrix),
                    input_length=50,
                    trainable=True))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.25))
model.add(Dense(units=2, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

batch_size = 128
history = model.fit(X_train, y_train, epochs=5, batch_size=batch_size, verbose=1, validation_split=0.1)

print(X_train.shape,y_train.shape)

# save the model to disk
filename = 'cyberbully_model.sav'
pickle.dump(model, open(filename, 'wb'))

y_hat = model.predict(X_test)
print(accuracy_score(list(map(lambda x: np.argmax(x), y_test)), list(map(lambda x: np.argmax(x), y_hat))))

# load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, y_test)
# print(result)