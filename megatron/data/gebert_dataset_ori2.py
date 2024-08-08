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
                 max_num_samples, encoder_seq_length, sentence_split_type,
                 masked_type, seed):

        # Params to store.
        self.name = name
        self.seed = seed
        self.encoder_seq_length = encoder_seq_length
        self.sentence_split_type = sentence_split_type
        self.masked_type = masked_type

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
        self.vocab_id_list = list(tokenizer.inv_vocab.keys())
        self.vocab_id_to_token_dict = tokenizer.inv_vocab
        self.pad_id = tokenizer.pad_id
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
        self.unk_id = tokenizer.unk_id
        self.mask_id = tokenizer.mask_id
        self.sentinel_tokens = tokenizer.sentinel_tokens_ids
        assert len(self.sentinel_tokens) > 0, "Provide the argument --vocab-extra-ids 100 to the script"

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
                                    self.sentence_split_type,
                                    self.masked_type,
                                    )
    
def build_training_sample(src_tokens, tgt_tokens, encoder_seq_length,
                        vocab_id_list, vocab_id_to_token_dict,
                        np_rng,
                        eos_token_id, bos_token_id, pad_token_id, unk_token_id, mask_token_id,
                        sentinel_tokens=None, 
                        sentence_split_type=None, 
                        masked_type=None):
    # import pdb; pdb.set_trace()
    if len(tgt_tokens) > 0:
        # import pdb; pdb.set_trace()
        if len(src_tokens) + len(tgt_tokens) > encoder_seq_length - 3:
            trun_index = encoder_seq_length - len(tgt_tokens) -3 
            src_tokens = src_tokens[:trun_index] 
        # source_masked_lm_prob = 0.15
        # source_max_predictions_per_seq = source_masked_lm_prob * len(src_tokens)
        # (source_masked_tokens, source_masked_positions, source_masked_labels, _, _) = bert_like_masking(
        #     src_tokens, vocab_id_list, vocab_id_to_token_dict, source_masked_lm_prob,
        #     eos_token_id, bos_token_id, pad_token_id, unk_token_id, mask_token_id, source_max_predictions_per_seq, np_rng)
        source_masked_tokens = src_tokens.tolist()
        source_masked_positions = []
        source_masked_labels = []

        target_masked_lm_prob = 0.3 + 0.4 * np_rng.rand()
        target_max_predictions_per_seq = target_masked_lm_prob * len(tgt_tokens)
        
        (target_masked_tokens, target_masked_positions, target_masked_labels, _, _) = span_uniform_masking(
        tgt_tokens, vocab_id_list, vocab_id_to_token_dict, target_masked_lm_prob,
        eos_token_id, bos_token_id, pad_token_id, unk_token_id, mask_token_id, target_max_predictions_per_seq, np_rng, max_ngrams=3, do_whole_word_mask=False, masking_style="t5")

        # (target_masked_tokens, target_masked_positions, target_masked_labels) = lazy_uniform_masking(tgt_tokens, mask_token_id, np_rng)

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
        # if len(src_tokens) + len(tgt_tokens) > encoder_seq_length:
        #     trun_index = encoder_seq_length - len(tgt_tokens)
        #     src_tokens = src_tokens[:trun_index] 
        src_tokens, tgt_tokens, attention_index = random_split(src_tokens, min_seq_length= 0.2 * len(src_tokens), np_rng = np_rng)

        source_masked_lm_prob = 0.15
        source_max_predictions_per_seq = source_masked_lm_prob * len(src_tokens)
        (source_masked_tokens, source_masked_positions, source_masked_labels, _, _) = bert_like_masking(
            src_tokens, vocab_id_list, vocab_id_to_token_dict, source_masked_lm_prob,
            eos_token_id, bos_token_id, pad_token_id, unk_token_id, mask_token_id, source_max_predictions_per_seq, np_rng, do_whole_word_mask=False, masking_style="t5")

        target_masked_lm_prob = np_rng.rand()
        target_max_predictions_per_seq = target_masked_lm_prob * len(tgt_tokens)

        (target_masked_tokens, target_masked_positions, target_masked_labels, _, _) = span_uniform_masking(
        tgt_tokens, vocab_id_list, vocab_id_to_token_dict, target_masked_lm_prob,
        eos_token_id, bos_token_id, pad_token_id, unk_token_id, mask_token_id, target_max_predictions_per_seq, np_rng, max_ngrams=3, do_whole_word_mask=False, masking_style="t5")

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

    index_mask = np.zeros((max_seq_length, max_seq_length))
    if attention_index is not None:
        index_mask[:attention_index, :attention_index]=1
        index_mask[attention_index:num_tokens, :num_tokens]=1
    
    index_mask_np = np.array(index_mask, dtype=np.int64)
    return tokens_np, labels_np, padding_mask_np, loss_mask_np, index_mask_np


# # Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

# """T5 Style dataset."""

# import collections

# import numpy as np
# import torch

# from megatron import get_tokenizer, print_rank_0
# from megatron.data.gebert_utils import (
#     get_samples_mapping,
#     get_finetune_samples_mapping,
#     random_split,
#     bert_like_masking,
#     span_uniform_masking,
#     lazy_uniform_masking 
# )

# class GeBertDataset(torch.utils.data.Dataset):

#     def __init__(self, name, indexed_dataset, data_prefix,
#                  max_num_samples, encoder_seq_length, sentence_split_type,
#                  masked_type, seed):

#         # Params to store.
#         self.name = name
#         self.seed = seed
#         self.encoder_seq_length = encoder_seq_length
#         self.sentence_split_type = sentence_split_type
#         self.masked_type = masked_type

#         # Dataset.
#         self.indexed_dataset = indexed_dataset

#         # Build the samples mapping.

#         #    new sample_mapping
#         # import pdb; pdb.set_trace()
#         self.type_mapping, self.samples_mapping = get_samples_mapping(self.indexed_dataset,
#                                                                     data_prefix, max_num_samples,
#                                                                       seed, name)
#         #---------------------------------------------------

#         # Vocab stuff.
#         tokenizer = get_tokenizer()
#         # import pdb; pdb.set_trace()
#         self.vocab_id_list = list(tokenizer.inv_vocab.keys())
#         self.vocab_id_to_token_dict = tokenizer.inv_vocab
#         self.pad_id = tokenizer.pad
#         self.bos_id = tokenizer.bos
#         self.eos_id = tokenizer.eos
#         self.unk_id = tokenizer.pad
#         self.mask_id = tokenizer.mask
#         self.sentinel_tokens = None
#         # assert len(self.sentinel_tokens) > 0, "Provide the argument --vocab-extra-ids 100 to the script"

#     def __len__(self):
#         return self.samples_mapping.shape[0]

#     def __getitem__(self, idx):
#         data_type = self.type_mapping[idx]
#         data_index = self.samples_mapping[idx]
#         src_tokens, tgt_tokens = self.indexed_dataset[(data_type, data_index)]
#         # Note that this rng state should be numpy and not python since
#         # python randint is inclusive whereas the numpy one is exclusive.
#         np_rng = np.random.RandomState(seed=((self.seed + idx) % 2**32))
#         return build_training_sample(src_tokens, tgt_tokens,
#                                     self.encoder_seq_length,
#                                     self.vocab_id_list,
#                                     self.vocab_id_to_token_dict,
#                                     np_rng,
#                                     self.eos_id, self.bos_id, self.pad_id, self.unk_id, self.mask_id,
#                                     self.sentinel_tokens,
#                                     self.sentence_split_type,
#                                     self.masked_type,
#                                     )
    
# def build_training_sample(src_tokens, tgt_tokens, encoder_seq_length,
#                         vocab_id_list, vocab_id_to_token_dict,
#                         np_rng,
#                         eos_token_id, bos_token_id, pad_token_id, unk_token_id, mask_token_id,
#                         sentinel_tokens=None, 
#                         sentence_split_type=None, 
#                         masked_type=None):
#     import pdb; pdb.set_trace()
#     if len(tgt_tokens) > 0:
#         # import pdb; pdb.set_trace()
#         if len(src_tokens) + len(tgt_tokens) > encoder_seq_length - 3:
#             trun_index = encoder_seq_length - len(tgt_tokens) -3 
#             src_tokens = src_tokens[:trun_index] 
#         # source_masked_lm_prob = 0.15
#         # source_max_predictions_per_seq = source_masked_lm_prob * len(src_tokens)
#         # (source_masked_tokens, source_masked_positions, source_masked_labels, _, _) = bert_like_masking(
#         #     src_tokens, vocab_id_list, vocab_id_to_token_dict, source_masked_lm_prob,
#         #     eos_token_id, bos_token_id, pad_token_id, unk_token_id, mask_token_id, source_max_predictions_per_seq, np_rng)
#         source_masked_tokens = src_tokens.tolist()
#         source_masked_positions = []
#         source_masked_labels = []

#         target_masked_lm_prob = 0.3 + 0.4 * np_rng.rand()
#         target_max_predictions_per_seq = target_masked_lm_prob * len(tgt_tokens)
        
#         (target_masked_tokens, target_masked_positions, target_masked_labels, _, _) = span_uniform_masking(
#         tgt_tokens, vocab_id_list, vocab_id_to_token_dict, target_masked_lm_prob,
#         eos_token_id, bos_token_id, pad_token_id, unk_token_id, mask_token_id, target_max_predictions_per_seq, np_rng, max_ngrams=3, do_whole_word_mask=False, masking_style="t5")

#         # (target_masked_tokens, target_masked_positions, target_masked_labels) = lazy_uniform_masking(tgt_tokens, mask_token_id, np_rng)

#         merge_tokens = [bos_token_id]+source_masked_tokens+[bos_token_id]+target_masked_tokens+[eos_token_id]
#         merge_masked_positions = [i+1 for i in source_masked_positions] + [i+len(source_masked_tokens)+2 for i in target_masked_positions]
#         merge_labels =  source_masked_labels + target_masked_labels

#         attention_index = len(src_tokens) + 1
#         tokens_np, labels_np, padding_mask_np, loss_mask_np, index_mask_np \
#             = pad_and_convert_to_numpy(merge_tokens, merge_masked_positions,
#                                     merge_labels, pad_token_id, encoder_seq_length, attention_index)      

#         train_sample = {
#             "text": tokens_np,
#             "labels": labels_np,
#             "loss_mask": loss_mask_np,
#             "padding_mask": padding_mask_np,
#             "index_mask": index_mask_np
#         }
#         return train_sample
#     else:
#         # if len(src_tokens) + len(tgt_tokens) > encoder_seq_length:
#         #     trun_index = encoder_seq_length - len(tgt_tokens)
#         #     src_tokens = src_tokens[:trun_index] 
#         src_tokens, tgt_tokens, attention_index = random_split(src_tokens, min_seq_length= 0.2 * len(src_tokens), np_rng = np_rng)

#         source_masked_lm_prob = 0.15
#         source_max_predictions_per_seq = source_masked_lm_prob * len(src_tokens)
#         (source_masked_tokens, source_masked_positions, source_masked_labels, _, _) = bert_like_masking(
#             src_tokens, vocab_id_list, vocab_id_to_token_dict, source_masked_lm_prob,
#             eos_token_id, bos_token_id, pad_token_id, unk_token_id, mask_token_id, source_max_predictions_per_seq, np_rng, do_whole_word_mask=False, masking_style="t5")

#         target_masked_lm_prob = np_rng.rand()
#         target_max_predictions_per_seq = target_masked_lm_prob * len(tgt_tokens)

#         (target_masked_tokens, target_masked_positions, target_masked_labels, _, _) = span_uniform_masking(
#         tgt_tokens, vocab_id_list, vocab_id_to_token_dict, target_masked_lm_prob,
#         eos_token_id, bos_token_id, pad_token_id, unk_token_id, mask_token_id, target_max_predictions_per_seq, np_rng, max_ngrams=3, do_whole_word_mask=False, masking_style="t5")

#         merge_tokens = [bos_token_id]+source_masked_tokens+target_masked_tokens
#         merge_masked_positions = [i+1 for i in source_masked_positions] + [i+len(source_masked_tokens)+1 for i in target_masked_positions]
#         merge_labels =  source_masked_labels + target_masked_labels

#         attention_index = attention_index + 1
#         tokens_np, labels_np, padding_mask_np, loss_mask_np, index_mask_np \
#             = pad_and_convert_to_numpy(merge_tokens, merge_masked_positions,
#                                     merge_labels, pad_token_id, encoder_seq_length, attention_index)      

#         train_sample = {
#             "text": tokens_np,
#             "labels": labels_np,
#             "loss_mask": loss_mask_np,
#             "padding_mask": padding_mask_np,
#             "index_mask": index_mask_np
#         }
#         return train_sample




# def pad_and_convert_to_numpy(tokens, masked_positions,
#                              masked_labels, pad_id, max_seq_length, attention_index=None):
#     """Pad sequences and convert them to numpy."""

#     # Some checks.
#     num_tokens = len(tokens)
#     padding_length = max_seq_length - num_tokens
#     assert padding_length >= 0, \
#         f"num_tokens ({num_tokens}) is greater than " \
#         "max_seq_length ({max_seq_length})."
#     assert len(masked_positions) == len(masked_labels)

#     # Tokens and token types.
#     filler = [pad_id] * padding_length
#     tokens_np = np.array(tokens + filler, dtype=np.int64)

#     # Padding mask.
#     padding_mask_np = np.array([1] * num_tokens + [0] * padding_length,
#                                dtype=np.int64)

#     # Lables and loss mask.
#     labels = [-1] * max_seq_length
#     loss_mask = [0] * max_seq_length
#     for i in range(len(masked_positions)):
#         assert masked_positions[i] < num_tokens
#         labels[masked_positions[i]] = masked_labels[i]
#         loss_mask[masked_positions[i]] = 1
#     labels_np = np.array(labels, dtype=np.int64)
#     loss_mask_np = np.array(loss_mask, dtype=np.int64)

#     index_mask = np.zeros((max_seq_length, max_seq_length))
#     if attention_index is not None:
#         index_mask[:attention_index, :attention_index]=1
#         index_mask[attention_index:num_tokens, :num_tokens]=1
    
#     index_mask_np = np.array(index_mask, dtype=np.int64)
#     return tokens_np, labels_np, padding_mask_np, loss_mask_np, index_mask_np


