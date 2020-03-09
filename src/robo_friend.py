# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 21:03:45 2020

@author: Aron Arceo
"""

import nltk
nltk.download('punkt' , quiet=True)
nltk.download('wordnet' , quiet=True)
import numpy as np
import random
import string # to process standard python strings


f=open('chatbot.txt','r',errors = 'ignore')
raw=f.read()
raw=raw.lower()# converts to lowercase

sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words

print(word_tokens[1])
print("Type of sent_tokens:({})".format(type(sent_tokens)))
print("Type of word_tokens:({})".format(type(word_tokens)))

lemmer = nltk.stem.WordNetLemmatizer()
#WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))