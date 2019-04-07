
import pandas as pd
import re
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from nltk.tokenize import  TweetTokenizer, word_tokenize
import gensim
import csv
from sacremoses import  MosesDetokenizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from get_top_words import filter_to_top_x


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

df['ans1'] = listLabel
counter = Counter(df['ans1'])
top_10_varieties = {i[0]: idx for idx, i in enumerate(counter.most_common(2))}
df = df[df['ans1'].map(lambda x: x in top_10_varieties)]

description_list = df['post'].tolist()
varietal_list = [top_10_varieties[i] for i in df['ans1'].tolist()]
varietal_list = pd.np.array(varietal_list)

count_vect = TfidfVectorizer(lowercase=True, tokenizer=lambda text: word_tokenize(text))
x_train_counts = count_vect.fit_transform(description_list)

tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

train_x, test_x, train_y, test_y = train_test_split(x_train_tfidf, varietal_list, test_size=0.1)

clf = OneVsRestClassifier(SVC(kernel='linear', gamma=0.5)).fit(train_x, train_y)
y_score = clf.predict(test_x)

n_right = 0
for i in range(len(y_score)):
    if y_score[i] == test_y[i]:
        n_right += 1

print("Accuracy: %.2f%%" % ((n_right / float(len(test_y)) * 100)))
print(classification_report(test_y, y_score))
