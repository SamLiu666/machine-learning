import nltk
from nltk.book import *
""""
s.startswith(t) Test if s starts with t
s.endswith(t) Test if s ends with t
t in s Test if t is contained inside s
s.islower() Test if all cased characters in s are lowercase
s.isupper() Test if all cased characters in s are uppercase
s.isalpha() Test if all characters in s are alphabetic
s.isalnum() Test if all characters in s are alphanumeric
s.isdigit() Test if all characters in s are digits
s.istitle() Test if s is titlecased (all words in s have initial capitals)
"""
def basic():
    # searching text
    print(text1.concordance("love"))
    print(text1.similar("love"))
    print(text2.common_contexts(["love","you"]))
    text4.dispersion_plot(["citizens", "democracy", "freedom", "duties", "America"])
    print(text2.generate("man"))


def lexical_diversity(text):
    return len(text) / len(set(text))


def percentage(count, total):
    return 100 * count / total


print(lexical_diversity(text1))
print(percentage(3, 9))

fdist1 = FreqDist(text1)

print(fdist1.values(), fdist1.keys())