import pandas
import os
import xlsxwriter
import normalise
from normalise.normalisation import normalise
import re

import re, string, unicodedata
import nltk
import contractions
import inflect
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

#Remove Noise
def ReplaceTagBR(text):
    text = text.replace("A:", "answerpost")
    text = text.replace("Q:", "questionpost")
    return text
def strip_html(text):
    text_new=""
    if(text != None and text != ""):
        soup = BeautifulSoup(text, "html.parser")
        text_new = soup.get_text()
    return text_new

def remove_url(text):
    return re.sub("(https?|http)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]", "", text)

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def to_lowercase(text):
    """Convert all characters to lowercase from list of tokenized words"""
    if(text != None and text != ""):
        text = text.lower()
    return text

def remove_punctuation(text):
    """Remove punctuation from list of tokenized words"""
    text = re.sub(r'[^\w\s]', ' ', text)
    return text


#denoise text
def denoise_text(text):
    text =  ReplaceTagBR(text)
    text = to_lowercase(text)
    text = strip_html(text)
    text = remove_url(text)
    text = remove_between_square_brackets(text)
    text = remove_punctuation(text)
    return text


#Normalization
def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words
def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

#normalize data
def normalize(words):
    words = replace_numbers(words)
    #words = remove_stopwords(words)
    return words

#search maximum severity
def maxseverity(df1, df2, df3):
    maks = df1
    indeks = 6
    if(df2 > maks):
        maks = df2
        indeks = 9
    if(df3 > maks):
        maks = df3
        indeks = 12
    return indeks
    
# Read csv file and normalization text
def read_csv_file_by_pandas(csv_file):
    data_frame = None
    if(os.path.exists(csv_file)):
        data_frame = pandas.read_csv(csv_file, sep='\t')
        df2 = data_frame.iloc[:, 0].str.split("\t", expand=True)
        for i in range(0, len(df2.index), 1):
            if(df2.iloc[i, 1] != None and  df2.iloc[i, 1]!=""):
                indeksmaks = 0
                #Normalize Severity
                if(df2.iloc[i, 6] == "None" or df2.iloc[i, 6] == None or df2.iloc[i, 6].isdigit() == False):
                    df2.iloc[i, 6] = 0
                if(df2.iloc[i, 9] == "None" or df2.iloc[i, 9] == None or df2.iloc[i, 9].isdigit() == False):
                    df2.iloc[i, 9] = 0
                if(df2.iloc[i, 12] == "None" or df2.iloc[i, 12] == None or df2.iloc[i, 12].isdigit() == False):
                    df2.iloc[i, 12] = 0

                #Classification    
                if(int(df2.iloc[i, 6]) != 0 or int(df2.iloc[i, 9]) != 0  or int(df2.iloc[i, 12]) != 0 ):
                    indeksmaks = maxseverity(int(df2.iloc[i, 6]), int(df2.iloc[i, 9]), int(df2.iloc[i, 12]) )
                    df2.iloc[i, 6] = int(df2.iloc[i, indeksmaks])
                    df2.iloc[i, 5] = "Bullying"
                    df2.iloc[i, 7] = df2.iloc[i, indeksmaks+1]
                else:
                    df2.iloc[i, 6] = 0
                    df2.iloc[i, 5] = "NotBullying"
                    df2.iloc[i, 7] = ""
                    
                df2.iloc[i, 1] = ' '.join(normalize(normalise(denoise_text(df2.iloc[i, 1]), verbose=False)))
                #df2.iloc[i, 7] = ' '.join(normalize(normalise(denoise_text(df2.iloc[i, 7]), verbose=False)))
            else:
                df2.iloc[i, 1] = "Empty"
                df2.iloc[i, 6] = 0
                df2.iloc[i, 5] = "NotBullying"
                df2.iloc[i, 7] = ""
        df2.columns = [n.replace('', '') for n in data_frame.columns.str.split('\t')[0]]
        df2.drop("ques", axis=1, inplace=True)
        df2.drop("ans", axis=1, inplace=True)
        df2.drop("userid", axis=1, inplace=True)
        df2.drop("asker", axis=1, inplace=True)
        df2.drop("ans2", axis=1, inplace=True)
        df2.drop("severity2", axis=1, inplace=True)
        df2.drop("bully2", axis=1, inplace=True)
        df2.drop("ans3", axis=1, inplace=True)
        df2.drop("severity3", axis=1, inplace=True)
        df2.drop("bully3", axis=1, inplace=True)
    else:
        print(csv_file + " do not exist.")    
    return df2

# Write csv file.
def write_to_csv_file_by_pandas(csv_file_path, data_frame):
    data_frame.to_csv(csv_file_path)
    print(csv_file_path + ' has been created.') 

def main():
    data_frame = read_csv_file_by_pandas("Small_TestingData.csv")
    write_to_csv_file_by_pandas("./Small_CleanData.csv", data_frame)
if __name__=='__main__':
    main()
    
