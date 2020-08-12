# train-pos.py
# A program to train a POS tagger
# 
# Run the program this way:
#   python3 train-pos.py train.en train.pos trans.json emiss.json
# 

import sys
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


def emission_helper(txt_corpus, pos_corpus, word, tag):
    pass


def train_pos_mll(txt_corpus, pos_corpus):
    words, tags = [], []  # list for save all the words and tags
    length = len(txt_corpus)
    for i in range(length):
    # Part 1. Complete the loop to collect the context needed to calculate emission prob. statistics
    # This can be the count or the probability, up to you.
    emission = {}

    # Part 2. Complete the loop to collect the context needed to calculate transition prob.statistics
    # This can be the count or the probability, up to you.
    transition = {}

    return emission, transition

if __name__ == '__main__':
    main()
