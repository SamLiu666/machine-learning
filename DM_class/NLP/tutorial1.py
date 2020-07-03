import stanza
import spacy
import pandas as pd
import numpy as np
from collections import Counter
from glob import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# The following are optional dependencies.
# Feel free to comment these out.
# Sent2tree uses the sent2tree.py module in this repository.
# from sent2tree import sentenceTree
# import ete3
import seaborn

# Display plots in this notebook, instead of externally.
from pylab import rcParams
rcParams['figure.figsize'] = 16, 8
# %matplotlib inline


def operation_stanza():
    #stanza.download('en') # download English model
    nlp = stanza.Pipeline('en') # initialize English neural pipeline
    doc = nlp("Barack Obama was born in Hawaii.")  # run annotation over a sentence
    print(doc)
    print(doc.entities)

print('##########################  Spacy')
# spacy.info()
nlp = spacy.load('en_core_web_lg')
# 读取文档
grail_raw = open('grail.txt', 'r', encoding="utf-8").read()
pride_raw = open('pride.txt', 'r', encoding="utf-8").read()
grail = nlp(grail_raw)
pride = nlp(pride_raw)
# print(pride[0], pride[:10], next(pride.sents))

def max_length_sentence():
    prideSents = list(pride.sents)
    # For example, let's find the longest sentence(s) in Pride and Prejudice:
    prideSentenceLength =[len(sent) for sent in prideSents]
    longestLength = max(prideSentenceLength)
    print("Pride 文本中最长的句子： ",longestLength)

def locations(needle, haystack):
    """list: """
    return pd.Series(np.histogram(
        [word.i for word in haystack
         if word.text.lower() == needle], bins=50)[0])


# rcParams['figure.figsize'] = 16, 8
# df = pd.DataFrame(
#     {name: locations(name.lower(), pride)
#      for name in ['Elizabeth', 'Darcy', 'Jane', 'Bennet']}
# ).plot(subplots=True)
# df.T.plot(kind='bar')

# entity recognition
entity_grail = set([w for w in grail.ents])
en = set([ent.string for ent in grail.ents if ent.label_ == 'NORP'])
print(entity_grail, '\n', en)

print('########################## Part of speech')
tagDict = {w.pos: w.pos_ for w in pride}
print(tagDict, '\n')

grailPOS = pd.Series(grail.count_by(spacy.attrs.POS))/len(grail)
pridePOS = pd.Series(pride.count_by(spacy.attrs.POS))/len(pride)

# rcParams['figure.figsize'] = 16, 8
# df = pd.DataFrame([grailPOS, pridePOS], index=['Grail', 'Pride'])
# df.columns = [tagDict[column] for column in df.columns]
# df.T.plot(kind='bar').show()

print('########################## Part of speech')
coconut, africanSwallow, europeanSwallow, horse = nlp('coconut'), nlp('African Swallow'), nlp('European Swallow'), nlp('horse')
print(coconut.similarity(horse), africanSwallow.similarity(horse), africanSwallow.similarity(horse))

prideNouns = [word for word in pride if word.pos_.startswith('N')][:150]
prideNounVecs = [word.vector for word in prideNouns]
prideNounLabels = [word.string.strip() for word in prideNouns]
lsa = TruncatedSVD(n_components=2)
lsaOut = lsa.fit_transform(prideNounVecs)
xs, ys = lsaOut[:,0], lsaOut[:,1]
for i in range(len(xs)):
    plt.scatter(xs[i], ys[i])
    plt.annotate(prideNounLabels[i], (xs[i], ys[i]))
plt.show()