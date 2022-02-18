import re
import nltk 
import pandas as pd
import numpy as np
from nltk import pos_tag # for parts of speech
from nltk import word_tokenize # to create tokens
from nltk.stem import wordnet # to perform lemmitization
from nltk.corpus import stopwords # for stop words
from sklearn.feature_extraction.text import CountVectorizer # to perform bow
from sklearn.metrics import pairwise_distances # to perfrom cosine similarity

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("emotions.csv", names = ["userplan", "emotion"])

def prepare(text):
    text = re.sub('[^ a-z]+', '', text)
    tokens = nltk.word_tokenize(text) # word tokenizing
    lemma = wordnet.WordNetLemmatizer() # intializing lemmatization
    tags_list = pos_tag(tokens, tagset=None)
    words = []   # empty list 
        
    for token, pos_token in tags_list:
        if token in set(stopwords.words('english')):
            continue
    
        if pos_token.startswith('V'):  # Verb
            pos_val='v'
        elif pos_token.startswith('J'): # Adjective
            pos_val='a'
        elif pos_token.startswith('R'): # Adverb
            pos_val='r'
        else:
            pos_val='n' # Noun
            
        lemma_token=lemma.lemmatize(token,pos_val) # performing lemmatization
        words.append(lemma_token) # appending the lemmatized token into a list
    
    return " ".join(words) # returns the lemmatized tokens as a sentence 
    
'''
df["userplan"] = df["userplan"].apply(prepare)
df.to_csv("emotions1.csv", index_label = None)
'''
cv = CountVectorizer() 
X = cv.fit_transform(df["userplan"]).toarray()
features = cv.get_feature_names()
df_bow = pd.DataFrame(X, columns = features)

def chat_bow(response):
    processed = prepare(response)
    bag = cv.transform([processed]).toarray() # applying bow
    cosine_value = 1 - pairwise_distances(df_bow, bag, metric = 'cosine' )
    index_value = cosine_value.argmax() # getting index value 
    return df["emotion"].loc[index_value]

while 1:
    userplan = input("Greetings! What are you up to? How are you feeling?\n")
    print("Playing " + chat_bow(userplan) + " music")

