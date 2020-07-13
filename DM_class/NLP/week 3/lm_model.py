from nltk.util import pad_sequence
from nltk.util import bigrams
from nltk.util import ngrams
from nltk.util import everygrams
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import flatten
import os
import requests
import io #codecs


text = [['a', 'b', 'c'], ['a', 'c', 'd', 'c', 'e', 'f']]
one = list(bigrams(text[0]))
two = list(pad_sequence(text[0],
                        pad_left=True, left_pad_symbol="<s>", pad_right=True, right_pad_symbol="</s>", n=2))
two = list(bigrams(two))
print(one,"\n", two)

with io.open('language-never-random.txt', encoding='utf8') as fin:
    text = fin.read()