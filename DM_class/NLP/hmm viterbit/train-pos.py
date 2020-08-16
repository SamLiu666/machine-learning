# train-pos.py
# A program to train a POS tagger
# 
# Run the program this way:
#   python3 train-pos.py train.en train.pos trans.json emiss.json
# 

import sys,time, nltk
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
                txt = ["<s>"] +  txt.lower().strip().split() + ["</s>"]
                pos = ["<s>"] + pos.strip().split() + ["</s>"]

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


def train_pos_mll(txt_corpus, pos_corpus):
    # preparation work
    start = time.time()
    #  1.dictionary: 'TAG':{word1: count(word1,'TAG')}
    train_word_tag = {}
    for sentence, tag_sentence in zip(txt_corpus, pos_corpus):
        for word, tag in zip(sentence, tag_sentence):
            try:
                try:
                    train_word_tag[tag][word] += 1
                except:
                    train_word_tag[tag][word] = 1
            except:
                train_word_tag[tag] = {word: 1}

    # 2. dictionary for transition probability
    bigram_tag_data = {}
    for tag_sentence in pos_corpus:
        bigram_ = list(nltk.bigrams(tag_sentence))
        for b1, b2 in bigram_:
            try:
                try:
                    bigram_tag_data[b1][b2] += 1
                except:
                    bigram_tag_data[b1][b2] = 1
            except:
                bigram_tag_data[b1] = {b2: 1}

    # Part 1. Complete the loop to collect the context needed to calculate emission prob. statistics
    # This can be the count or the probability, up to you.
    emission = {}
    for key in train_word_tag.keys():
        emission[key] = {}
        count = sum(train_word_tag[key].values())
        for key_word in train_word_tag[key].keys():
            emission[key][key_word] = train_word_tag[key][key_word]/count

    # Part 2. Complete the loop to collect the context needed to calculate transition prob.statistics
    # This can be the count or the probability, up to you.
    transition = {}
    # almost the same as part2
    for k in bigram_tag_data.keys():
        transition[k] = {}
        count = sum(bigram_tag_data[k].values())
        for k2 in bigram_tag_data[k].keys():
            transition[k][k2] = bigram_tag_data[k][k2] / count

    end = time.time()
    print("time cost:  ", end-start)
    return emission, transition


if __name__ == '__main__':
    main()
