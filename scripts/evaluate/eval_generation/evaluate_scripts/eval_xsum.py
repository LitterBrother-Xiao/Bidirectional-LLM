# pip install pycocoevalcap
# check https://github.com/pltrdy/files2rouge to install files2rouge


#!/usr/bin/env python
from __future__ import print_function
__author__ = 'xinya'

from collections import defaultdict
from argparse import ArgumentParser
import string
import os

import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

_tok_dict = {"(": "-lrb-", ")": "-rrb-",
             "[": "-lsb-", "]": "-rsb-",
             "{": "-lcb-", "}": "-rcb-",
             "[UNK]": "UNK", '&': '&amp;', '<': '&lt;', '>': '&gt;'}


def _is_digit(w):
    for ch in w:
        if not(ch.isdigit() or ch == ','):
            return False
    return True


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def fix_tokenization(text):
    input_tokens = text.split()
    output_tokens = []
    has_left_quote = False
    has_left_single_quote = False

    i = 0
    prev_dash = False
    while i < len(input_tokens):
        tok = input_tokens[i]
        flag_prev_dash = False
        if tok in _tok_dict.keys():
            output_tokens.append(_tok_dict[tok])
            i += 1
        elif tok == "\"":
            if has_left_quote:
                output_tokens.append("''")
            else:
                output_tokens.append("``")
            has_left_quote = not has_left_quote
            i += 1
        elif tok == "'" and len(output_tokens) > 0 and output_tokens[-1].endswith("n") and i < len(input_tokens) - 1 and input_tokens[i + 1] == "t":
            output_tokens[-1] = output_tokens[-1][:-1]
            output_tokens.append("n't")
            i += 2
        elif tok == "'" and i < len(input_tokens) - 1 and input_tokens[i + 1] in ("s", "d", "ll"):
            output_tokens.append("'"+input_tokens[i + 1])
            i += 2
        elif tok == "'":
            if has_left_single_quote:
                output_tokens.append("'")
            else:
                output_tokens.append("`")
            has_left_single_quote = not has_left_single_quote
            i += 1
        elif tok == "." and i < len(input_tokens) - 2 and input_tokens[i + 1] == "." and input_tokens[i + 2] == ".":
            output_tokens.append("...")
            i += 3
        elif tok == "," and len(output_tokens) > 0 and _is_digit(output_tokens[-1]) and i < len(input_tokens) - 1 and _is_digit(input_tokens[i + 1]):
            # $ 3 , 000 -> $ 3,000
            output_tokens[-1] += ','+input_tokens[i + 1]
            i += 2
        elif tok == "." and len(output_tokens) > 0 and output_tokens[-1].isdigit() and i < len(input_tokens) - 1 and input_tokens[i + 1].isdigit():
            # 3 . 03 -> $ 3.03
            output_tokens[-1] += '.'+input_tokens[i + 1]
            i += 2
        elif tok == "." and len(output_tokens) > 0 and len(output_tokens[-1]) == 1 and output_tokens[-1].isupper() and i < len(input_tokens) - 2 and len(input_tokens[i + 1]) == 1 and input_tokens[i + 1].isupper() and input_tokens[i + 2] == '.':
            # U . N . -> U.N.
            k = i+3
            while k+2 < len(input_tokens):
                if len(input_tokens[k + 1]) == 1 and input_tokens[k + 1].isupper() and input_tokens[k + 2] == '.':
                    k += 2
                else:
                    break
            output_tokens[-1] += ''.join(input_tokens[i:k])
            i += 2
        elif tok == "-":
            if i < len(input_tokens) - 1 and input_tokens[i + 1] == "-":
                output_tokens.append("--")
                i += 2
            elif i == len(input_tokens) - 1 or i == 0:
                output_tokens.append("-")
                i += 1
            elif output_tokens[-1] not in string.punctuation and input_tokens[i + 1][0] not in string.punctuation:
                output_tokens[-1] += "-"
                i += 1
                flag_prev_dash = True
            else:
                output_tokens.append("-")
                i += 1
        elif prev_dash and len(output_tokens) > 0 and tok[0] not in string.punctuation:
            output_tokens[-1] += tok
            i += 1
        else:
            output_tokens.append(tok)
            i += 1
        prev_dash = flag_prev_dash
    return " ".join(output_tokens)

def fix_tokenization_xsum(text):
    input_tokens = text.replace(" ##", "").strip().split()
    output_tokens = []
    has_left_quote = False
    has_left_single_quote = False

    i = 0
    prev_dash = False
    while i < len(input_tokens):
        tok = input_tokens[i]
        flag_prev_dash = False
        if tok == "'" and len(output_tokens) > 0 and output_tokens[-1].endswith("n") and i < len(input_tokens) - 1 and input_tokens[i + 1] == "t":
            output_tokens[-1] = output_tokens[-1][:-1]
            output_tokens[-1] += "n't"
            i += 2
        elif tok == "'" and i < len(input_tokens) - 1 and input_tokens[i + 1] in ("s", "d", "ll", "re"):
            if len(output_tokens) > 0:
                output_tokens[-1] += "'"+input_tokens[i + 1]
            else:
                output_tokens.append("'"+input_tokens[i + 1])
            i += 2
        elif tok == "." and i < len(input_tokens) - 2 and input_tokens[i + 1] == "." and input_tokens[i + 2] == ".":
            output_tokens.append("...")
            i += 3
        elif tok == "," and len(output_tokens) > 0 and _is_digit(output_tokens[-1]) and i < len(input_tokens) - 1 and _is_digit(input_tokens[i + 1]):
            # $ 3 , 000 -> $ 3,000
            output_tokens[-1] += ','+input_tokens[i + 1]
            i += 2
        elif tok == "." and len(output_tokens) > 0 and output_tokens[-1].isdigit() and i < len(input_tokens) - 1 and input_tokens[i + 1].isdigit():
            # 3 . 03 -> $ 3.03
            output_tokens[-1] += '.'+input_tokens[i + 1]
            i += 2
        elif tok == "." and len(output_tokens) > 0 and len(output_tokens[-1]) == 1 and output_tokens[-1].isupper() and i < len(input_tokens) - 2 and len(input_tokens[i + 1]) == 1 and input_tokens[i + 1].isupper() and input_tokens[i + 2] == '.':
            # U . N . -> U.N.
            k = i+3
            while k+2 < len(input_tokens):
                if len(input_tokens[k + 1]) == 1 and input_tokens[k + 1].isupper() and input_tokens[k + 2] == '.':
                    k += 2
                else:
                    break
            output_tokens[-1] += ''.join(input_tokens[i:k])
            i += 2
        elif tok == "-":
            if i < len(input_tokens) - 1 and input_tokens[i + 1] == "-":
                output_tokens.append("--")
                i += 2
            elif i == len(input_tokens) - 1 or i == 0:
                output_tokens.append("-")
                i += 1
            elif output_tokens[-1] not in string.punctuation and input_tokens[i + 1][0] not in string.punctuation:
                output_tokens[-1] += "-"
                i += 1
                flag_prev_dash = True
            else:
                output_tokens.append("-")
                i += 1
        elif prev_dash and len(output_tokens) > 0 and tok[0] not in string.punctuation:
            output_tokens[-1] += tok
            i += 1
        else:
            output_tokens.append(tok)
            i += 1
        prev_dash = flag_prev_dash
    return " ".join(output_tokens)

def eval(out_file, src_file, tgt_file):
    """
        Given a filename, calculate the metric scores for that prediction file
        isDin: boolean value to check whether input file is DirectIn.txt
    """

    pairs = []
    with open(src_file, 'r') as infile:
        for line in infile:
            pair = {}
            pair['tokenized_sentence'] = line[:-1].strip().lower()
            pairs.append(pair)


    reflen = 0
    with open(tgt_file, "r") as infile:
        cnt = 0
        for line in infile:
            pairs[cnt]['tokenized_question'] = line[:-1].strip()
            reflen += len(pairs[cnt]['tokenized_question'].split())
            cnt += 1

    genlen = 0
    output = []
    with open(out_file, 'r') as infile:
        for line in infile:
            # if fix_token:
            line = fix_tokenization(line[:-1].strip()).lower()
            genlen += len(line.split())
            # else:
            #     line = line[:-1].strip().lower()
            output.append(line)

    # with open(".xsum_output.txt", "w") as f:
    #     for line in output:
    #         f.write(line + "\n")
    # return os.system(f"files2rouge .xsum_output.txt {tgt_file}")
    print(f"genlen: {genlen}, reflen: {reflen}, ratio: {genlen / reflen}")


#python eval.py --out '/Users/royokong/Downloads/ckpt-32000.dev' \
     #--src '/Users/royokong/nlp/unilm-nat/data/squadqg_data/org_data/dev.src'\
     #--tgt '/Users/royokong/nlp/unilm-nat/data/squadqg_data/org_data/dev.tgt'
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-out", "--out_file", dest="out_file", help="output file to compare",
                        default="xxx.out")
    parser.add_argument("-src", "--src_file", dest="src_file",
                        default="xxx.src", help="src file")
    parser.add_argument("-tgt", "--tgt_file", dest="tgt_file",
                        default="xxx.tgt", help="target file")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    eval(args.out_file, args.src_file, args.tgt_file)

    print("scores: \n")

    if args.fast:
        os.system(f"files2rouge {args.out_file} {args.tgt_file} --ignore_empty_reference -a \"-c 95 -r 10 -n 2 -a\"")
    else:
        os.system(f"files2rouge {args.out_file} {args.tgt_file} --ignore_empty_reference")