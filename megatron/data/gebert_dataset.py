# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""T5 Style dataset."""

import collections

import numpy as np
import torch

from megatron import get_tokenizer, print_rank_0
from megatron.data.gebert_utils import (
    get_samples_mapping,
    get_finetune_samples_mapping,
    random_split,
    bert_like_masking,
    span_uniform_masking,
    lazy_uniform_masking 
)

class GeBertDataset(torch.utils.data.Dataset):

    def __init__(self, name, indexed_dataset, data_prefix,
                 max_num_samples, encoder_seq_length, has_sentence_split,
                 masked_type, masked_x_type, has_attention_masking, seed):

        # Params to store.
        self.name = name
        self.seed = seed
        self.encoder_seq_length = encoder_seq_length
        self.has_sentence_split = has_sentence_split
        self.masked_type = masked_type
        self.masked_x_type = masked_x_type
        self.has_attention_masking = has_attention_masking

        # Dataset.
        self.indexed_dataset = indexed_dataset

        # Build the samples mapping.

        #    new sample_mapping
        # import pdb; pdb.set_trace()
        self.type_mapping, self.samples_mapping = get_samples_mapping(self.indexed_dataset,
                                                                    data_prefix, max_num_samples,
                                                                      seed, name)
        #---------------------------------------------------

        # Vocab stuff.
        tokenizer = get_tokenizer()
        # import pdb; pdb.set_trace()
        self.vocab_id_list = list(tokenizer.inv_vocab.keys())
        self.vocab_id_to_token_dict = tokenizer.inv_vocab
        self.pad_id = tokenizer.pad
        self.bos_id = tokenizer.bos
        self.eos_id = tokenizer.eos
        self.unk_id = tokenizer.pad
        self.mask_id = tokenizer.mask
        self.sentinel_tokens = None
        # assert len(self.sentinel_tokens) > 0, "Provide the argument --vocab-extra-ids 100 to the script"

    def __len__(self):
        return self.samples_mapping.shape[0]

    def __getitem__(self, idx):
        data_type = self.type_mapping[idx]
        data_index = self.samples_mapping[idx]
        src_tokens, tgt_tokens = self.indexed_dataset[(data_type, data_index)]
        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        np_rng = np.random.RandomState(seed=((self.seed + idx) % 2**32))
        return build_training_sample(src_tokens, tgt_tokens,
                                    self.encoder_seq_length,
                                    self.vocab_id_list,
                                    self.vocab_id_to_token_dict,
                                    np_rng,
                                    self.eos_id, self.bos_id, self.pad_id, self.unk_id, self.mask_id,
                                    self.sentinel_tokens,
                                    self.has_sentence_split,
                                    self.masked_type,
                                    self.masked_x_type,
                                    self.has_attention_masking
                                    )
    
def build_training_sample(src_tokens, tgt_tokens, encoder_seq_length,
                        vocab_id_list, vocab_id_to_token_dict,
                        np_rng,
                        eos_token_id, bos_token_id, pad_token_id, unk_token_id, mask_token_id,
                        sentinel_tokens=None, 
                        has_sentence_split=False, 
                        masked_type=None,
                        masked_x_type=None,
                        has_attention_masking=False):
    # import pdb; pdb.set_trace()
    if len(tgt_tokens) > 0:
        # import pdb; pdb.set_trace()
        if len(src_tokens) + len(tgt_tokens) > encoder_seq_length - 3:
            trun_index = encoder_seq_length - len(tgt_tokens) -3 
            src_tokens = src_tokens[:trun_index] 

        target_masked_lm_prob = np_rng.rand()
        target_max_predictions_per_seq = target_masked_lm_prob * len(tgt_tokens)
        
        (target_masked_tokens, target_masked_positions, target_masked_labels, _, _) = span_uniform_masking(
        tgt_tokens, vocab_id_list, vocab_id_to_token_dict, target_masked_lm_prob,
        eos_token_id, bos_token_id, pad_token_id, unk_token_id, mask_token_id, target_max_predictions_per_seq, np_rng, max_ngrams=3, do_whole_word_mask=False, masking_style="t5")

        if masked_x_type == 'adaptive':
            source_masked_lm_prob = 0.3 - target_masked_lm_prob * 0.2
            source_max_predictions_per_seq = max(1,source_masked_lm_prob * len(single_src_tokens))
            (source_masked_tokens, _, _, _, _) = bert_like_masking(
                single_src_tokens, vocab_id_list, vocab_id_to_token_dict, source_masked_lm_prob,
                eos_token_id, bos_token_id, pad_token_id, unk_token_id, mask_token_id, source_max_predictions_per_seq, np_rng, do_whole_word_mask=False, masking_style="t5")
        elif masked_x_type == 'no_mask':
            source_masked_tokens = src_tokens.tolist()
            source_masked_positions = []
            source_masked_labels = []

        merge_tokens = [bos_token_id]+source_masked_tokens+[bos_token_id]+target_masked_tokens+[eos_token_id]
        merge_masked_positions = [i+1 for i in source_masked_positions] + [i+len(source_masked_tokens)+2 for i in target_masked_positions]
        merge_labels =  source_masked_labels + target_masked_labels

        attention_index = len(src_tokens) + 1
        tokens_np, labels_np, padding_mask_np, loss_mask_np, index_mask_np \
            = pad_and_convert_to_numpy(merge_tokens, merge_masked_positions,
                                    merge_labels, pad_token_id, encoder_seq_length, attention_index)      

        train_sample = {
            "text": tokens_np,
            "labels": labels_np,
            "loss_mask": loss_mask_np,
            "padding_mask": padding_mask_np,
            "index_mask": index_mask_np
        }
        return train_sample
    else:
        if not has_sentence_split and not has_attention_masking:
            bos_index_list = np.argwhere(src_tokens == bos_token_id).tolist()
            eos_index_list = np.argwhere(src_tokens == eos_token_id).tolist()   
            if len(eos_index_list) == 0:
                if len(bos_index_list) == 0:
                    append_bos = False
                else:
                    append_bos = True
                    assert src_tokens[0] == bos_token_id
                    src_tokens = src_tokens[1:]
                
                src_tokens = src_tokens.tolist()
                mask_lm_prob = 0.15
                max_predictions_per_seq = max(1, mask_lm_prob * len(src_tokens))
                masked_tokens, masked_positions, masked_labels, _ , _ = span_uniform_masking(
                    src_tokens, vocab_id_list, vocab_id_to_token_dict, mask_lm_prob,
                    eos_token_id, bos_token_id, pad_token_id, unk_token_id, mask_token_id, max_predictions_per_seq, np_rng, max_ngrams=3, do_whole_word_mask=False, masking_style="t5")

                if append_bos:
                    masked_tokens = [bos_token_id] + masked_tokens
                    masked_positions = [i+1 for i in masked_positions]

                tokens_np, labels_np, padding_mask_np, loss_mask_np, index_mask_np \
                    = pad_and_convert_to_numpy(masked_tokens, masked_positions,
                                            masked_labels, pad_token_id, encoder_seq_length)      

                train_sample = {
                    "text": tokens_np,
                    "labels": labels_np,
                    "loss_mask": loss_mask_np,
                    "padding_mask": padding_mask_np,
                }
                return train_sample
            
            else:
                merge_tokens_all = []
                merge_labels_all = []
                merge_masked_positions_all = []
                source_attention_index_list = []
                target_attention_index_list = []
                source_index = 0
                target_index = 0
                for index in range(len(eos_index_list)+1):
                    # import pdb; pdb.set_trace()
                    append_bos = False
                    append_eos = False
                    if index == 0:
                        single_sample = src_tokens[:eos_index_list[index][0]+1]
                    elif index == len(eos_index_list):
                        single_sample = src_tokens[eos_index_list[index-1][0]+1:]
                    else:
                        single_sample = src_tokens[eos_index_list[index-1][0]+1:eos_index_list[index][0]+1]

                    if len(single_sample) < 12:
                        merge_tokens_all.extend(single_sample)
                        target_index = len(merge_tokens_all)
                    else:
                        if single_sample[0] == bos_token_id:
                            single_sample = single_sample[1:]
                            append_bos = True
                        if single_sample[-1] == eos_token_id:
                            single_sample = single_sample[:-1]
                            append_eos = True

                        single_sample = single_sample.tolist()
                        mask_lm_prob = 0.15
                        max_predictions_per_seq = max(1, mask_lm_prob * len(single_sample))
                        masked_tokens, masked_positions, masked_labels, _, _ = span_uniform_masking(
                            single_sample, vocab_id_list, vocab_id_to_token_dict, mask_lm_prob,
                            eos_token_id, bos_token_id, pad_token_id, unk_token_id, mask_token_id, max_predictions_per_seq, np_rng, max_ngrams=3, do_whole_word_mask=False, masking_style="t5")

                        if (append_bos) and (not append_eos):
                            masked_tokens = [bos_token_id] + masked_tokens 
                            masked_positions = [i+1 for i in masked_positions]
                        
                        elif (not append_bos) and (append_eos):
                            masked_tokens = masked_tokens + [eos_token_id] 
                        
                        elif (append_bos) and (append_eos):
                            masked_tokens = [bos_token_id] + masked_tokens + [eos_token_id]
                            masked_positions = [i+1 for i in masked_positions]

                        merge_masked_positions_all.extend(i+target_index for i in masked_positions)
                        merge_tokens_all.extend(masked_tokens)
                        merge_labels_all.extend(masked_labels)
                        target_index = len(merge_tokens_all)

                assert len(merge_tokens_all) == encoder_seq_length
                assert len(merge_labels_all) == len(merge_masked_positions_all)     
                tokens_np, labels_np, padding_mask_np, loss_mask_np, index_mask_np \
                    = pad_and_convert_to_numpy(masked_tokens, masked_positions,
                                            masked_labels, pad_token_id, encoder_seq_length)      
                train_sample = {
                    "text": tokens_np,
                    "labels": labels_np,
                    "loss_mask": loss_mask_np,
                    "padding_mask": padding_mask_np,
                }
                return train_sample

        
        elif has_sentence_split and has_attention_masking:
            
            bos_index_list = np.argwhere(src_tokens == bos_token_id).tolist()
            eos_index_list = np.argwhere(src_tokens == eos_token_id).tolist()   
            if len(eos_index_list) == 0:
                if len(bos_index_list) == 0:
                    append_bos = False
                else:
                    append_bos = True
                    assert src_tokens[0] == bos_token_id
                    src_tokens = src_tokens[1:]
                    
                single_src_tokens, single_tgt_tokens, attention_index = random_split(src_tokens, min_seq_length= 0.2 * len(src_tokens), np_rng = np_rng)

                target_masked_lm_prob = np_rng.rand()
                target_max_predictions_per_seq = max(1,target_masked_lm_prob * len(single_tgt_tokens))

                (target_masked_tokens, target_masked_positions, target_masked_labels, _, _) = span_uniform_masking(
                    single_tgt_tokens, vocab_id_list, vocab_id_to_token_dict, target_masked_lm_prob,
                    eos_token_id, bos_token_id, pad_token_id, unk_token_id, mask_token_id, target_max_predictions_per_seq, np_rng, max_ngrams=3, do_whole_word_mask=False, masking_style="t5")

                if masked_x_type == 'adaptive':
                    # print('ssss')
                    source_masked_lm_prob = 0.3 - target_masked_lm_prob * 0.2
                    source_max_predictions_per_seq = max(1,source_masked_lm_prob * len(single_src_tokens))
                    (source_masked_tokens, source_masked_positions, source_masked_labels, _, _) = bert_like_masking(
                        single_src_tokens, vocab_id_list, vocab_id_to_token_dict, source_masked_lm_prob,
                        eos_token_id, bos_token_id, pad_token_id, unk_token_id, mask_token_id, source_max_predictions_per_seq, np_rng, do_whole_word_mask=False, masking_style="t5")
                elif masked_x_type == 'random':
                    source_masked_lm_prob = 0.15
                    source_max_predictions_per_seq = source_masked_lm_prob * len(single_src_tokens)
                    (source_masked_tokens, source_masked_positions, source_masked_labels, _, _) = bert_like_masking(
                        single_src_tokens, vocab_id_list, vocab_id_to_token_dict, source_masked_lm_prob,
                        eos_token_id, bos_token_id, pad_token_id, unk_token_id, mask_token_id, source_max_predictions_per_seq, np_rng, do_whole_word_mask=False, masking_style="t5")
                elif masked_x_type == 'nomask':
                    source_masked_tokens = src_tokens.tolist()
                    source_masked_positions = []
                    source_masked_labels = []
                
                if not append_bos:
                    merge_tokens = source_masked_tokens+target_masked_tokens
                    merge_masked_positions = source_masked_positions + [i+len(source_masked_tokens) for i in target_masked_positions]
                    merge_labels =  source_masked_labels + target_masked_labels
                else:
                    merge_tokens = [bos_token_id]+source_masked_tokens+target_masked_tokens
                    merge_masked_positions = [i+1 for i in source_masked_positions] + [i+len(source_masked_tokens)+1 for i in target_masked_positions]
                    merge_labels =  source_masked_labels + target_masked_labels
                    attention_index = attention_index + 1

                tokens_np, labels_np, padding_mask_np, loss_mask_np, index_mask_np \
                    = pad_and_convert_to_numpy(merge_tokens, merge_masked_positions,
                                            merge_labels, pad_token_id, encoder_seq_length, attention_index)      

                train_sample = {
                    "text": tokens_np,
                    "labels": labels_np,
                    "loss_mask": loss_mask_np,
                    "padding_mask": padding_mask_np,
                    "index_mask": index_mask_np
                }
                return train_sample
                
            else:
                merge_tokens_all = []
                merge_labels_all = []
                merge_masked_positions_all = []
                source_attention_index_list = []
                target_attention_index_list = []
                source_index = 0
                target_index = 0
                for index in range(len(eos_index_list)+1):
                    append_bos = False
                    append_eos = False
                    if index == 0:
                        single_sample = src_tokens[:eos_index_list[index][0]+1]
                    elif index == len(eos_index_list):
                        single_sample = src_tokens[eos_index_list[index-1][0]+1:]
                    else:
                        single_sample = src_tokens[eos_index_list[index-1][0]+1:eos_index_list[index][0]+1]

                    if len(single_sample) < 12:
                        merge_tokens_all.extend(single_sample)
                        target_index = len(merge_tokens_all)
                        source_attention_index_list.append(target_index)
                        target_attention_index_list.append(target_index)
                    else:
                        if single_sample[0] == bos_token_id:
                            single_sample = single_sample[1:]
                            append_bos = True
                        if single_sample[-1] == eos_token_id:
                            single_sample = single_sample[:-1]
                            append_eos = True
                        
                        single_src_tokens, single_tgt_tokens, attention_index = random_split(single_sample, min_seq_length= 0.2 * len(single_sample), np_rng = np_rng)

                        target_masked_lm_prob = np_rng.rand()
                        target_max_predictions_per_seq = max(1,target_masked_lm_prob * len(single_tgt_tokens))
                        (target_masked_tokens, target_masked_positions, target_masked_labels, _, _) = span_uniform_masking(
                        single_tgt_tokens, vocab_id_list, vocab_id_to_token_dict, target_masked_lm_prob,
                        eos_token_id, bos_token_id, pad_token_id, unk_token_id, mask_token_id, target_max_predictions_per_seq, np_rng, max_ngrams=3, do_whole_word_mask=False, masking_style="t5")

                        if masked_x_type =='adaptive':
                            source_masked_lm_prob = 0.3 - 0.2 * target_masked_lm_prob
                            source_max_predictions_per_seq = max(1,source_masked_lm_prob * len(single_src_tokens))
                            (source_masked_tokens, source_masked_positions, source_masked_labels, _, _) = bert_like_masking(
                                single_src_tokens, vocab_id_list, vocab_id_to_token_dict, source_masked_lm_prob,
                                eos_token_id, bos_token_id, pad_token_id, unk_token_id, mask_token_id, source_max_predictions_per_seq, np_rng, do_whole_word_mask=False, masking_style="t5")
                        elif masked_x_type =='random':
                            source_masked_lm_prob = 0.15
                            source_max_predictions_per_seq = max(1,source_masked_lm_prob * len(single_src_tokens))
                            (source_masked_tokens, source_masked_positions, source_masked_labels, _, _) = bert_like_masking(
                                single_src_tokens, vocab_id_list, vocab_id_to_token_dict, source_masked_lm_prob,
                                eos_token_id, bos_token_id, pad_token_id, unk_token_id, mask_token_id, source_max_predictions_per_seq, np_rng, do_whole_word_mask=False, masking_style="t5")
                        elif masked_x_type=='no_mask':
                            single_src_tokens = single_src_tokens.tolist()
                            source_masked_positions = []
                            source_masked_labels = []

                        if (not append_bos) and (not append_eos):
                            merge_tokens = source_masked_tokens+target_masked_tokens
                            merge_masked_positions = source_masked_positions + [i+len(source_masked_tokens) for i in target_masked_positions]
                            merge_labels =  source_masked_labels + target_masked_labels

                        elif (append_bos) and (not append_eos):
                            merge_tokens = [bos_token_id]+source_masked_tokens+target_masked_tokens
                            merge_masked_positions = [i+1 for i in source_masked_positions] + [i+len(source_masked_tokens)+1 for i in target_masked_positions]
                            merge_labels =  source_masked_labels + target_masked_labels
                            attention_index = attention_index + 1
                        
                        elif (not append_bos) and (append_eos):
                            merge_tokens = source_masked_tokens+target_masked_tokens+[eos_token_id]
                            merge_masked_positions = source_masked_positions + [i+len(source_masked_tokens) for i in target_masked_positions]
                            merge_labels =  source_masked_labels + target_masked_labels
                        
                        elif (append_bos) and (append_eos):
                            merge_tokens = [bos_token_id]+source_masked_tokens+target_masked_tokens+[eos_token_id]
                            merge_masked_positions = [i+1 for i in source_masked_positions] + [i+len(source_masked_tokens)+1 for i in target_masked_positions]
                            merge_labels =  source_masked_labels + target_masked_labels
                            attention_index = attention_index + 1
                        
                        merge_masked_positions_all.extend(i+target_index for i in merge_masked_positions)
                        merge_tokens_all.extend(merge_tokens)
                        merge_labels_all.extend(merge_labels)
                        source_index = target_index + attention_index
                        target_index = len(merge_tokens_all)
                        source_attention_index_list.append(source_index)
                        target_attention_index_list.append(target_index)

                assert len(merge_tokens_all) == encoder_seq_length
                assert len(merge_labels_all) == len(merge_masked_positions_all)
                tokens_np, labels_np, padding_mask_np, loss_mask_np, index_mask_np \
                    = pad_and_convert_to_numpy_group(merge_tokens_all, merge_masked_positions_all,
                                            merge_labels_all, pad_token_id, encoder_seq_length, source_attention_index_list, target_attention_index_list)
                assert len(tokens_np) == encoder_seq_length
                train_sample = {
                    "text": tokens_np,
                    "labels": labels_np,
                    "loss_mask": loss_mask_np,
                    "padding_mask": padding_mask_np,
                    "index_mask": index_mask_np
                }
                return train_sample

        elif has_sentence_split and not has_attention_masking:
            # import pdb; pdb.set_trace()
            bos_index_list = np.argwhere(src_tokens == bos_token_id).tolist()
            eos_index_list = np.argwhere(src_tokens == eos_token_id).tolist()   
            if len(eos_index_list) == 0:
                if len(bos_index_list) == 0:
                    append_bos = False
                else:
                    append_bos = True
                    assert src_tokens[0] == bos_token_id
                    src_tokens = src_tokens[1:]
                    
                single_src_tokens, single_tgt_tokens, attention_index = random_split(src_tokens, min_seq_length= 0.2 * len(src_tokens), np_rng = np_rng)

                target_masked_lm_prob = np_rng.rand()
                target_max_predictions_per_seq = max(1,target_masked_lm_prob * len(single_tgt_tokens))

                (target_masked_tokens, target_masked_positions, target_masked_labels, _, _) = span_uniform_masking(
                    single_tgt_tokens, vocab_id_list, vocab_id_to_token_dict, target_masked_lm_prob,
                    eos_token_id, bos_token_id, pad_token_id, unk_token_id, mask_token_id, target_max_predictions_per_seq, np_rng, max_ngrams=3, do_whole_word_mask=False, masking_style="t5")
                
                if masked_x_type == 'random':
                    source_masked_lm_prob = 0.15
                    source_max_predictions_per_seq = max(1,source_masked_lm_prob * len(single_src_tokens))
                    (source_masked_tokens, source_masked_positions, source_masked_labels, _, _) = bert_like_masking(
                        single_src_tokens, vocab_id_list, vocab_id_to_token_dict, source_masked_lm_prob,
                        eos_token_id, bos_token_id, pad_token_id, unk_token_id, mask_token_id, source_max_predictions_per_seq, np_rng, do_whole_word_mask=False, masking_style="t5")
                elif masked_x_type =='adaptive':
                    source_masked_lm_prob = 0.3 - 0.2 * target_masked_lm_prob 
                    source_max_predictions_per_seq = max(1,source_masked_lm_prob * len(single_src_tokens))
                    (source_masked_tokens, source_masked_positions, source_masked_labels, _, _) = bert_like_masking(
                        single_src_tokens, vocab_id_list, vocab_id_to_token_dict, source_masked_lm_prob,
                        eos_token_id, bos_token_id, pad_token_id, unk_token_id, mask_token_id, source_max_predictions_per_seq, np_rng, do_whole_word_mask=False, masking_style="t5")
                elif masked_x_type =='nomask':
                    single_src_tokens = single_src_tokens.tolist()
                    source_masked_positions = []
                    source_masked_labels = []

                if not append_bos:
                    merge_tokens = source_masked_tokens+target_masked_tokens
                    merge_masked_positions = source_masked_positions + [i+len(source_masked_tokens) for i in target_masked_positions]
                    merge_labels =  source_masked_labels + target_masked_labels
                else:
                    merge_tokens = [bos_token_id]+source_masked_tokens+target_masked_tokens
                    merge_masked_positions = [i+1 for i in source_masked_positions] + [i+len(source_masked_tokens)+1 for i in target_masked_positions]
                    merge_labels =  source_masked_labels + target_masked_labels
                    attention_index = attention_index + 1

                tokens_np, labels_np, padding_mask_np, loss_mask_np, index_mask_np \
                    = pad_and_convert_to_numpy(merge_tokens, merge_masked_positions,
                                            merge_labels, pad_token_id, encoder_seq_length)      

                train_sample = {
                    "text": tokens_np,
                    "labels": labels_np,
                    "loss_mask": loss_mask_np,
                    "padding_mask": padding_mask_np,
                }
                return train_sample
                
            else:
                merge_tokens_all = []
                merge_labels_all = []
                merge_masked_positions_all = []
                source_attention_index_list = []
                target_attention_index_list = []
                source_index = 0
                target_index = 0
                for index in range(len(eos_index_list)+1):
                    # import pdb; pdb.set_trace()
                    append_bos = False
                    append_eos = False
                    if index == 0:
                        single_sample = src_tokens[:eos_index_list[index][0]+1]
                    elif index == len(eos_index_list):
                        single_sample = src_tokens[eos_index_list[index-1][0]+1:]
                    else:
                        single_sample = src_tokens[eos_index_list[index-1][0]+1:eos_index_list[index][0]+1]

                    if len(single_sample) < 12:
                        merge_tokens_all.extend(single_sample)
                        target_index = len(merge_tokens_all)
                        source_attention_index_list.append(target_index)
                        target_attention_index_list.append(target_index)
                    else:
                        if single_sample[0] == bos_token_id:
                            single_sample = single_sample[1:]
                            append_bos = True
                        if single_sample[-1] == eos_token_id:
                            single_sample = single_sample[:-1]
                            append_eos = True
                        
                        single_src_tokens, single_tgt_tokens, attention_index = random_split(single_sample, min_seq_length= 0.2 * len(single_sample), np_rng = np_rng)

                        target_masked_lm_prob = np_rng.rand()
                        target_max_predictions_per_seq = max(1,target_masked_lm_prob * len(single_tgt_tokens))

                        (target_masked_tokens, target_masked_positions, target_masked_labels, _, _) = span_uniform_masking(
                        single_tgt_tokens, vocab_id_list, vocab_id_to_token_dict, target_masked_lm_prob,
                        eos_token_id, bos_token_id, pad_token_id, unk_token_id, mask_token_id, target_max_predictions_per_seq, np_rng, max_ngrams=3, do_whole_word_mask=False, masking_style="t5")
                        
                        if masked_x_type == 'random':
                            source_masked_lm_prob = 0.15
                            source_max_predictions_per_seq = max(1,source_masked_lm_prob * len(single_src_tokens))
                            (source_masked_tokens, source_masked_positions, source_masked_labels, _, _) = bert_like_masking(
                                single_src_tokens, vocab_id_list, vocab_id_to_token_dict, source_masked_lm_prob,
                                eos_token_id, bos_token_id, pad_token_id, unk_token_id, mask_token_id, source_max_predictions_per_seq, np_rng, do_whole_word_mask=False, masking_style="t5")
                        elif masked_x_type =='adaptive':
                            source_masked_lm_prob = 0.3 - 0.2 * target_masked_lm_prob 
                            source_max_predictions_per_seq = max(1,source_masked_lm_prob * len(single_src_tokens))
                            (source_masked_tokens, source_masked_positions, source_masked_labels, _, _) = bert_like_masking(
                                single_src_tokens, vocab_id_list, vocab_id_to_token_dict, source_masked_lm_prob,
                                eos_token_id, bos_token_id, pad_token_id, unk_token_id, mask_token_id, source_max_predictions_per_seq, np_rng, do_whole_word_mask=False, masking_style="t5")
                        elif masked_x_type =='nomask':
                            single_src_tokens = single_src_tokens.tolist()
                            source_masked_positions = []
                            source_masked_labels = []

                        if (not append_bos) and (not append_eos):
                            merge_tokens = source_masked_tokens+target_masked_tokens
                            merge_masked_positions = source_masked_positions + [i+len(source_masked_tokens) for i in target_masked_positions]
                            merge_labels =  source_masked_labels + target_masked_labels

                        elif (append_bos) and (not append_eos):
                            merge_tokens = [bos_token_id]+source_masked_tokens+target_masked_tokens
                            merge_masked_positions = [i+1 for i in source_masked_positions] + [i+len(source_masked_tokens)+1 for i in target_masked_positions]
                            merge_labels =  source_masked_labels + target_masked_labels
                            attention_index = attention_index + 1
                        
                        elif (not append_bos) and (append_eos):
                            merge_tokens = source_masked_tokens+target_masked_tokens+[eos_token_id]
                            merge_masked_positions = source_masked_positions + [i+len(source_masked_tokens) for i in target_masked_positions]
                            merge_labels =  source_masked_labels + target_masked_labels
                        
                        elif (append_bos) and (append_eos):
                            merge_tokens = [bos_token_id]+source_masked_tokens+target_masked_tokens+[eos_token_id]
                            merge_masked_positions = [i+1 for i in source_masked_positions] + [i+len(source_masked_tokens)+1 for i in target_masked_positions]
                            merge_labels =  source_masked_labels + target_masked_labels
                            attention_index = attention_index + 1
                        
                        merge_masked_positions_all.extend(i+target_index for i in merge_masked_positions)
                        # import pdb; pdb.set_trace()
                        merge_tokens_all.extend(merge_tokens)
                        merge_labels_all.extend(merge_labels)
                        source_index = target_index + attention_index
                        target_index = len(merge_tokens_all)
                        source_attention_index_list.append(source_index)
                        target_attention_index_list.append(target_index)

                assert len(merge_tokens_all) == encoder_seq_length
                assert len(merge_labels_all) == len(merge_masked_positions_all)
                tokens_np, labels_np, padding_mask_np, loss_mask_np, index_mask_np \
                    = pad_and_convert_to_numpy_group(merge_tokens_all, merge_masked_positions_all,
                                            merge_labels_all, pad_token_id, encoder_seq_length, source_attention_index_list=None, target_attention_index_list=None)
                assert len(tokens_np) == encoder_seq_length
                train_sample = {
                    "text": tokens_np,
                    "labels": labels_np,
                    "loss_mask": loss_mask_np,
                    "padding_mask": padding_mask_np,
                }
                return train_sample


def pad_and_convert_to_numpy(tokens, masked_positions,
                             masked_labels, pad_id, max_seq_length, attention_index=None):
    """Pad sequences and convert them to numpy."""

    # Some checks.
    num_tokens = len(tokens)
    padding_length = max_seq_length - num_tokens
    assert padding_length >= 0, \
        f"num_tokens ({num_tokens}) is greater than " \
        "max_seq_length ({max_seq_length})."
    assert len(masked_positions) == len(masked_labels)

    # Tokens and token types.
    filler = [pad_id] * padding_length
    tokens_np = np.array(tokens + filler, dtype=np.int64)

    # Padding mask.
    padding_mask_np = np.array([1] * num_tokens + [0] * padding_length,
                               dtype=np.int64)

    # Lables and loss mask.
    labels = [-1] * max_seq_length
    loss_mask = [0] * max_seq_length
    for i in range(len(masked_positions)):
        assert masked_positions[i] < num_tokens
        labels[masked_positions[i]] = masked_labels[i]
        loss_mask[masked_positions[i]] = 1
    labels_np = np.array(labels, dtype=np.int64)
    loss_mask_np = np.array(loss_mask, dtype=np.int64)

    index_mask_np = None
    if attention_index is not None:
        index_mask = np.zeros((max_seq_length, max_seq_length))
        index_mask[:attention_index, :attention_index]=1
        index_mask[attention_index:num_tokens, :num_tokens]=1
        # ----------------------------
        # no attaention index
        # index_mask[:num_tokens, :num_tokens]=1
        index_mask_np = np.array(index_mask, dtype=np.int64)
    
    return tokens_np, labels_np, padding_mask_np, loss_mask_np, index_mask_np


def pad_and_convert_to_numpy_group(tokens, masked_positions,
                             masked_labels, pad_id, max_seq_length, source_attention_index_list, target_attention_index_list):
    """Pad sequences and convert them to numpy."""

    # Some checks.
    num_tokens = len(tokens)
    padding_length = max_seq_length - num_tokens
    assert padding_length >= 0, \
        f"num_tokens ({num_tokens}) is greater than " \
        "max_seq_length ({max_seq_length})."
    assert len(masked_positions) == len(masked_labels)

    # Tokens and token types.
    filler = [pad_id] * padding_length
    tokens_np = np.array(tokens + filler, dtype=np.int64)

    # Padding mask.
    padding_mask_np = np.array([1] * num_tokens + [0] * padding_length,
                               dtype=np.int64)

    # Lables and loss mask.
    labels = [-1] * max_seq_length
    loss_mask = [0] * max_seq_length
    for i in range(len(masked_positions)):
        # assert masked_positions[i] < num_tokens
        labels[masked_positions[i]] = masked_labels[i]
        loss_mask[masked_positions[i]] = 1
    labels_np = np.array(labels, dtype=np.int64)
    loss_mask_np = np.array(loss_mask, dtype=np.int64)

    current_index = 0
    index_mask_np = None
    if source_attention_index_list is not None:
        index_mask = np.zeros((max_seq_length, max_seq_length))
        for index in range(len(source_attention_index_list)):
            source_index = source_attention_index_list[index]
            target_index = target_attention_index_list[index]
            index_mask[current_index:source_index, current_index:source_index]=1
            index_mask[source_index:target_index, current_index:target_index]=1
            current_index = target_index
        index_mask_np = np.array(index_mask, dtype=np.int64)
    
    return tokens_np, labels_np, padding_mask_np, loss_mask_np, index_mask_np





























