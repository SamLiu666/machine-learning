# tagger.py 
# A file to perform viterbi inference for POS tagging
# Use a file this way:
#    python3 tagger.py [trans_file] [emiss_file] < test.en > test.hyp

import sys
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trans_file")
    parser.add_argument("emiss_file")
    args = parser.parse_args()

    with open(args.trans_file) as trans_fp:
        trans = json.load(trans_fp)
    with open(args.emiss_file) as emiss_fp:
        emiss = json.load(emiss_fp)

    # First you need to collect all the possible tag set from the emission file
    tag_set = collect_tag_set(emiss)
    print(tag_set)
    # print(emiss)

    for line in sys.stdin:
        line = line.strip().lower().split()
        pos = viterbi_inference(trans, emiss, tag_set, line)
        print(" ".join(pos))


def collect_tag_set(emiss):
    tag_set = set()
    # Complete this part to collect all possible tag set
    for key, value in emiss.items():
        s = key.split(" ")   # fron train-pos: save as tag word
        tag_set.add(s[0])    # so take the first as tag
    return tag_set


def viterbi_inference(trans, emiss, tag_set, inp):
    # trans format: "tag pretag" ; emiss format: "tag word"
    pos_tag = []   #

    tag_list = list(tag_set)  # to use index
    words = set()   # vacab list
    for key, value in emiss.items():
        s = key.split(" ")   # fron train-pos: save as tag word
        words.add(s[1])
    print(len(words), len(tag_set))

    # Part 3. Complete this method to calculate pos tag using viterbi algorithm!
    for i in range(len(inp)):
        p_state = []   #  save for state probability
        prob_trainsition = []  # save for transition probability

        for tag in tag_list:
            # loop all the tag in tag list
            if i == 0:
                temp = " ".join([tag, "``"])  # start
            else:
                temp = " ".join([tag, pos_tag[-1]])  # not start

            tran_prob = trans[temp]  # use the tag key to get probability
            print(temp, tran_prob)

            # compute emission and state probabilities
            if inp[i] in words:
                emiss_prob = emiss[" ".join([tag, inp[i]])]  # use its original prob
            else:
                emiss_prob = tran_prob   # use traansition prob if word not in vocabulary
            # emiss_prob = emiss[" ".join([tag, inp[i]])]  # use its original prob
            state_probability = emiss_prob*tran_prob   # state prob
            p_state.append(state_probability)     # save in p_state list for choosing
            prob_trainsition.append(tran_prob)    # save for choosing tag

        p_max = max(p_state)

        if p_max == 0:
            # if unknow word appear, we use transition prob
            p_max = max(prob_trainsition)
            state_max = tag_list[prob_trainsition.index(p_max)]  # choose the coresonding tag
        else:
            state_max = tag_list[p_state.index(p_max)]

        pos_tag.append(state_max)

    return pos_tag


if __name__ == '__main__':
    main()
