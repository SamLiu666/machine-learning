# train-pos.py
# A program to train a POS tagger
# 
# Run the program this way:
#   python3 train-pos.py train.en train.pos trans.json emiss.json
# 

import sys,time
import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_txt")
    parser.add_argument("train_pos")
    parser.add_argument("trans_file_out")
    parser.add_argument("emiss_file_out")
    args = parser.parse_args()

    # reading input file & pos tag
    txt_corpus = []
    pos_corpus = []

    with open(args.train_txt, encoding="utf-8") as txt_fp:
        with open(args.train_pos, encoding="utf-8") as pos_fp:
            for txt, pos in zip(txt_fp, pos_fp):
                txt = txt.lower().strip().split() + ["</s>"]
                pos = pos.strip().split() + ["</s>"]

                assert len(txt) == len(pos)

                txt_corpus.append(txt)
                pos_corpus.append(pos)

    emission, transition = train_pos_mll(txt_corpus, pos_corpus)

    # Hint, to save a "tuple" key in json you can turn them into string
    # (a, b) -> "a b"
    with open(args.trans_file_out, "w") as tfp:
        json.dump(transition, fp=tfp, indent=4, sort_keys=True)

    with open(args.emiss_file_out, "w") as efp:
        json.dump(emission, fp=efp, indent=4, sort_keys=True)


def emission_helper(txt_corpus, pos_corpus, word, tag, tags_dict, words):
    em_count = 0.0
    V = len(words)
    #print(tag, word)
    for i in range(len(txt_corpus)):
        for j in range(len(txt_corpus[i])):
            if word == txt_corpus[i][j] and tag == pos_corpus[i][j]:
                em_count += 1.0
                #print(word, "->", tag)
    emiss_prob = (em_count+V)/(tags_dict[tag]+V)
    return emiss_prob


def transition_helper(pos_corpus, pre_tag, tag, tags_dict, words):
    tr_count = 0.0
    V = len(words)
    for i in range(len(pos_corpus)):
        for j in range(1, len(pos_corpus[i])):
            if pre_tag==pos_corpus[i][j-1] and tag==pos_corpus[i][j]:
                tr_count += 1.0
    trans_prob = (tr_count+V)/(tags_dict[tag]+V)
    return trans_prob


def train_pos_mll(txt_corpus, pos_corpus):
    # preparation work
    start = time.time()

    words, tags, tags_dict = [], [], {}  # list for save all the words and tags, tags dict for tag number
    length = len(txt_corpus)

    for i in range(length):
        for j in range(len(txt_corpus[i])):
            if txt_corpus[i][j] not in words:
                words.append(txt_corpus[i][j])   # list of all the word

            if pos_corpus[i][j] not in tags:
                tags.append(pos_corpus[i][j])    # list of all the tags

            if pos_corpus[i][j] not in tags_dict:
                tags_dict[pos_corpus[i][j]] = 1
            else:
                tags_dict[pos_corpus[i][j]] += 1   # dict for tags' count

    words.sort()
    tags.sort()

    row = len(tags)  # matrix for saving probabilities
    col = len(words)
    print(row, col)

    # Part 1. Complete the loop to collect the context needed to calculate emission prob. statistics
    # This can be the count or the probability, up to you.
    emission = {}
    for i in range(row):
        tag = tags[i]  # row for tag
        if tag == "</s>":
            continue   # because it not in test file
        print("processing: ", i+1, "/", row)
        for j in range(col):
            word = words[j]  # col for words
            temp = " ".join([tag, word])
            # compute the prob
            emission[temp] = emission_helper(txt_corpus, pos_corpus, word=word, tag=tag,
                                             tags_dict=tags_dict, words=words)
            if j%2000==0:
                print("emission: ", emission[temp])

    # Part 2. Complete the loop to collect the context needed to calculate transition prob.statistics
    # This can be the count or the probability, up to you.
    transition = {}
    for i in range(row):
        tag = tags[i]  # row for tag
        if tag == "</s>":
            continue
        print("processing: ", i + 1, "/", row)
        for j in range(row):
            pre_tag = tags[j]  # col for words
            temp = " ".join([tag, pre_tag])
            # compute the prob
            transition[temp] = transition_helper(pos_corpus,pre_tag=pre_tag, tag=tag,
                                                 tags_dict=tags_dict, words=words)
            if j%20==0:
                print("transition: ", transition[temp])
    end = time.time()
    print("time cost:  ", end-start)
    return emission, transition


if __name__ == '__main__':
    main()
