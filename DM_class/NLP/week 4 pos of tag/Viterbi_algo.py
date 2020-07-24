# Importing the required libraries
import nltk
import numpy as np
import itertools
from nltk.corpus import brown


##########################################################################################
sent_tag = brown.tagged_sents()
# print(sent_tag)  # corpus with position os tagging
mod_sent_tag = []
for s in sent_tag:
    s.insert(0, ("##", "##"))
    s.append(("&&", "&&"))
    mod_sent_tag.append(s)

#print("Corpus comparation:\n",mod_sent_tag[0], "\n", sent_tag[0])

##########################################################################################
# spliting the data for train and test
split_num = int(len(mod_sent_tag)*0.9)
train_data = mod_sent_tag[:split_num]
test_data = mod_sent_tag[split_num:]

##########################################################################################
#Creating a dictionary whose keys are tags and values contain words which were assigned the correspoding tag
# ex:- 'TAG':{word1: count(word1,'TAG')}
train_word_tag = {}
for s in train_data:
    for (w,t) in s:
        w = w.lower()
        try:
            try:
                train_word_tag[t][w] += 1
            except:
                train_word_tag[t][w] = 1
        except:
            train_word_tag[t] = {w:1}

##########################################################################################
#Calculating the emission probabilities using train_word_tag
train_emission_prob={}
for k in train_word_tag.keys():
  train_emission_prob[k]={}
  count = sum(train_word_tag[k].values())
  for k2 in train_word_tag[k].keys():
    train_emission_prob[k][k2]=train_word_tag[k][k2]/count