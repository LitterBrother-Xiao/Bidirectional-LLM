# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Processing data for pretraining."""

import argparse
import json
import multiprocessing
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import time

import torch
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False
    
from megatron.data import indexed_dataset
from megatron.tokenizer import build_tokenizer

from datasets import load_dataset

class Encoder(object):    
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)
        Encoder.group_size = self.args.group_size

    def encode(self, json_lines):
        concat_sentence_ids = []
        sentence_ids = None
        tot_len = 0
        for json_line in json_lines:
            data = json.loads(json_line)
            text_column = self.args.json_key
            text = data[text_column]
            sentence_ids = Encoder.tokenizer.tokenize(text)
            if len(sentence_ids) > 20:
                sentence_ids_new = [Encoder.tokenizer.bos] + sentence_ids + [Encoder.tokenizer.eos]
                concat_sentence_ids.extend(sentence_ids_new)
                tot_len += len(json_line)
        # group text
        group_size = Encoder.group_size
        total_length = len(concat_sentence_ids) // group_size * group_size
        group_text = [concat_sentence_ids[i : i + group_size] for i in range(0, total_length, group_size)]
        return group_text, tot_len

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--json-file', type=str, required=True, help='Path to input JSON')
    group.add_argument('--json-key', type=str, default='text')
    group.add_argument("--group-size", type=int, required=True)
    group.add_argument('--file-type', type=str, default='jsonl or parquet')


    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-model', type=str, required=True)
    group.add_argument('--vocab_extra_ids', type=int, default=0)
    group.add_argument('--tokenizer-type', type=str, default=None)
    group.add_argument('--vocab', type=str, default=None)

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True, help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap', choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, required=True, help='Number of worker processes to launch')
    group.add_argument('--batch-size', type=int, required=True)
    group.add_argument('--chunk-size', type=int, required=True, help='Chunk size assigned to each worker process')
    group.add_argument('--log-interval', type=int, default=100, help='Interval between progress updates')
    args = parser.parse_args()
    # args.keep_empty = False

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    # args.tokenizer_type = "SentencePieceTokenizer"
    # args.vocab_extra_ids = 0

    return args

def main():
    args = get_args()
    startup_start = time.time()
    
    output_idx_file_new = "{}.idx".format(args.output_prefix)
    if os.path.exists(output_idx_file_new):
        exit()

    print("Opening", args.json_file)
    fin = open(args.json_file, 'r', encoding='utf-8')

    if args.file_type=='parquet':
        single_dataset = load_dataset('parquet', data_files=args.json_file)

    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    # import pdb; pdb.set_trace()
    def group_iter(line_iter, batch_size):
        group_text = []
        for line in line_iter:
            group_text.append(line)
            if len(group_text) == batch_size:
                yield group_text
                group_text.clear()
        yield group_text
    
    def group_iter_parquet(single_dataset, batch_size):
        group_text = []
        for index in range(len(single_dataset)):
           current_text = single_dataset['train'][index]
           group_text.append(current_text)
           if len(group_text) == batch_size:
                yield group_text
                group_text.clear()
        yield group_text 
    
    if args.file_type=='parquet':
        encoded_docs_batch = pool.imap(encoder.encode, group_iter_parquet(single_dataset, args.batch_size), args.chunk_size)
    else:
        encoded_docs_batch = pool.imap(encoder.encode, group_iter(fin, args.batch_size), args.chunk_size)

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")
    output_bin_file = "{}.bin".format(args.output_prefix)
    output_idx_file = "{}.idx".format(args.output_prefix)
    builder = indexed_dataset.make_builder(output_bin_file,
                                            impl=args.dataset_impl,
                                            vocab_size=tokenizer.vocab_size)

    startup_end = time.time()
    proc_start = time.time()
    total_docs_processed, total_sentence_created, total_bytes_processed = 0, 0, 0
    print("Time to startup:", startup_end - startup_start)
    # count = 0
    # min_len = 1024
    for doc, bytes_processed in encoded_docs_batch:
        for sentence in doc:
            builder.add_item(
                source_tensor=torch.IntTensor(sentence), 
                target_tensor=torch.IntTensor([]),
                task="span-corruption",
            )
        total_docs_processed += args.batch_size
        total_sentence_created += len(doc)
        total_bytes_processed += bytes_processed
        if total_docs_processed % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"{total_docs_processed} docs, {total_docs_processed / elapsed:.0f} docs/s | ",
                  f"{total_sentence_created} sents, {total_sentence_created / elapsed:.0f} sents/s | ",
                  f"{mbs:.2f} MB/s | ")
    print("Done! Now finalizing.")
    builder.finalize(output_idx_file)

if __name__ == '__main__':
    main()





