# eval.py
# A file to calculate Precisision, Recall, and F1 score.
# Use a file this way:
#    python3 eval.py test.hyp test.pos

import sys
import json
import argparse
from collections import Counter
from tagger import viterbi_inference

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hyp_file")
    parser.add_argument("ref_file")
    args = parser.parse_args()

    # Calculate accuracy
    tp = 0
    total = 0
    with open(args.hyp_file) as hyp_fp:
        with open(args.ref_file) as ref_fp:
            for hyp, ref in zip(hyp_fp, ref_fp):
                hyp = hyp.strip().split()
                ref = ref.strip().split()

                assert len(hyp) == len(ref)
                #..... complete me ! ---
                for h,r in zip(hyp, ref):
                    if h == r:
                        tp += 1
                    total += 1
    print("Accuracy :", tp / total, file=sys.stderr)


if __name__ == '__main__':
    main()
