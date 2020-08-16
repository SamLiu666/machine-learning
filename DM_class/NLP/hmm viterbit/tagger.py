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
    parser.add_argument("test_txt")
    parser.add_argument("test_hyp")
    args = parser.parse_args()

    with open(args.trans_file) as trans_fp:
        trans = json.load(trans_fp)
    with open(args.emiss_file) as emiss_fp:
        emiss = json.load(emiss_fp)

    # First you need to collect all the possible tag set from the emission file
    tag_set = collect_tag_set(emiss)
    # generate the tag of test file
    f = open("test.hyp", "w", encoding="utf-8")
    with open(args.test_txt, encoding="utf-8") as txt_fp:
        for txt in txt_fp:
            txt = txt.lower().strip().split()
            pos = viterbi_inference(trans, emiss, tag_set, txt)
            # print(txt, "\n", pos)
            pos = " ".join(pos)
            f.write(pos)
            f.write("\n")
    f.close()


    # print(tag_set)
    # print(emiss)
    print("done!")
    for line in sys.stdin:
        line = line.strip().lower().split()
        pos = viterbi_inference(trans, emiss, tag_set, line)
        print(" ".join(pos))


def collect_tag_set(emiss):
    tag_set = {}
    # Complete this part to collect all possible tag set
    for k, v in emiss.items():
        for word, prob in v.items():
            try:
                if k not in tag_set[word]:
                    tag_set[word].append(k)
            except:
                temp = []
                temp.append(k)
                tag_set[word] = temp
    return tag_set


def viterbi_inference(trans, emiss, tag_set, inp):
    pos_tag = []
    # Part 3. Complete this method to calculate pos tag using viterbi algorithm!
    # {step_no.:{state1:[previous_best_state,value_of_the_state]}}
    final_state_value = {}
    inp.insert(0, '<s>')  # for the start
    inp.append('</s>')    # and the end

    for i in range(len(inp)):
        w = inp[i]  # word
        # the beginning
        if i == 1:
            final_state_value[i] = {}
            try:
                tags_in_step = tag_set[w]
            except:
                tags_in_step = list(set(list(trans.keys())))  # loop all the tags

            # to find the best tag prob
            for tag_step in tags_in_step:
                try:
                    # from the start, in the word list
                    final_state_value[i][tag_step] = ['<s>', trans['<s>'][tag_step] * emiss[tag_step][w]]
                except:
                    # if not in vocab list, assign 0.0001
                    final_state_value[i][tag_step] = ['<s>', 0.0001]

        # not the beginning
        if i > 1:
            final_state_value[i] = {}
            pre_state = list(final_state_value[i - 1].keys())  # previous state list for looping
            try:
                cur_state = tag_set[w]
            except:
                cur_state = list(set(list(trans.keys())))

            # loop the current state
            for t in cur_state:
                temp_state = []
                for pre_s in pre_state:
                    # almost the same logic as i==0
                    try:
                        temp_state.append(final_state_value[i - 1][pre_s][1] * trans[pre_s][t] * emiss[t][w])
                    except:
                        temp_state.append(final_state_value[i - 1][pre_s][1] * 0.0001)

                # find the best state based on the previous operation
                max_index = temp_state.index(max(temp_state))
                best_pre_state = pre_state[max_index]
                final_state_value[i][t] = [best_pre_state, max(temp_state)]

    # find the final tag sequence
    final_steps = final_state_value.keys()
    last_step = max(final_steps)
    for f_s in range(len(final_steps)):
        step = last_step - f_s
        if step == last_step:
            pos_tag.append('</s>')
            pos_tag.append(final_state_value[step]['</s>'][0])
        if step < last_step and step > 0:
            pos_tag.append(final_state_value[step][pos_tag[len(pos_tag) - 1]][0])
    pos_tag = list(reversed(pos_tag))
    pos_tag = pos_tag[1:-1]
    return pos_tag


if __name__ == '__main__':
    main()
