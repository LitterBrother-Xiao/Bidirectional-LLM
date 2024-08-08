

"""Pretrain GeBERT for generation tasks"""

from functools import partial

import torch
import torch.nn.functional as F

from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.gebert_utils import build_train_valid_test_datasets
from megatron.model import GeBertModel
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group
import argparse
from megatron.arguments import core_transformer_config_from_args

import random
from megatron import get_tokenizer
# from fairseq.utils import new_arange
def args_provider(parser):
    group = parser.add_argument_group(title='Extra args')
    # data preprocessing args
    group.add_argument('--has-sentence-split', action='store_true', help='If split the training sentence into X and Y')
    group.add_argument('--masked-type', type=str, default='src_bertlike_tgt_uniform', help='The masking type.')
    group.add_argument('--masked-x-type', type=str, default='adaptive', help='The source masking type.')
    group.add_argument('--has-attention-masking', action='store_true', help='If adopt the attention masking.')


    # length prediction args
    group.add_argument('--length-predict', action='store_true', help='if we adopt length prediction')
    group.add_argument('--max-predict-length', type=int, help='the maximun predicted length')
    group.add_argument('--length-factor', type=float, default=0.1, help='length loss factor')
    group.add_argument('--load-LP-module', action='store_true', help='if we load length prediction')

    # dpo args
    group.add_argument('--dpo-training', action='store_true', help='if we adopt dpo training')
    group.add_argument('--dpo-update-model-step', type=int, default=0, help='dpo update steps')
    group.add_argument('--dpo-sampling-type', type=str, default='random', help='How to sample the DPO samples.')
    group.add_argument('--dpo-type', type=str, default='dpo_mix', help='dpo_type')
    group.add_argument('--lambda_1', type=float, default=0.5, help='dpo loss factor1')
    group.add_argument('--lambda_2', type=float, default=10.0, help='dpo loss factor2')
    group.add_argument('--beta', type=float, default=0.1, help='dpo loss factor3')

    return parser
    
def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GEBERT model ...')

    args = get_args()
    config = core_transformer_config_from_args(args)
    model = GeBertModel(
            config=config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process)
    return model

def get_batch(data_iterator):
    """Build the batch."""
    args = get_args()
    # Items and their type.
    keys = ['text', 'labels', 'loss_mask', 'padding_mask']
    if args.has_attention_masking:
        keys.append('index_mask')
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens = data_b['text'].long()
    loss_mask = data_b['loss_mask'].float()
    lm_labels = data_b['labels'].long()
    padding_mask = data_b['padding_mask'].long()
    
    # import pdb; pdb.set_trace()
    if args.has_attention_masking:
        index_mask = data_b['index_mask'].long() 
        return tokens, loss_mask, lm_labels, padding_mask, index_mask
    else:
        return tokens, loss_mask, lm_labels, padding_mask

def loss_func(loss_mask, length_loss, chosen_reward, reject_reward, pos_loss, neg_loss, output_tensors=None):
    args = get_args()
    lm_loss_ = output_tensors.float()
    loss_mask = loss_mask.float()
    lm_loss = torch.sum(
        lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()
    loss = lm_loss
    averaged_losses = average_losses_across_data_parallel_group(
        [lm_loss])
    log_dict = {'lm loss': averaged_losses[0]}
    
    if length_loss is not None:
        loss+=length_loss
        log_dict['length loss'] = length_loss
    
    if args.dpo_training:
        if args.dpo_type == 'dpo_only':
            log_dict['chosen reward'] = chosen_reward
            log_dict['reject reward'] = reject_reward
            log_dict['pos_loss'] = pos_loss
            log_dict['neg_loss'] = neg_loss
            policy_loss = - F.logsigmoid(args.beta * (chosen_reward - reject_reward))
            log_dict['policy loss'] = policy_loss
            
            loss = args.lambda_1 * policy_loss + length_loss

        if args.dpo_type == 'ori_dpo':
            log_dict['chosen reward'] = chosen_reward
            log_dict['reject reward'] = reject_reward
            log_dict['pos_loss'] = pos_loss
            log_dict['neg_loss'] = neg_loss
            policy_loss = - F.logsigmoid(args.beta * (chosen_reward - reject_reward))
            log_dict['policy loss'] = policy_loss

            loss += args.lambda_1 * policy_loss
        elif args.dpo_type == 'smung':
            log_dict['chosen_reward'] = chosen_reward
            log_dict['reject_reward'] = reject_reward
            log_dict['pos_loss'] = pos_loss
            log_dict['neg_loss'] = neg_loss

            policy_loss = - F.logsigmoid(args.beta * (chosen_reward - reject_reward - args.lambda_2 * pos_loss))
            log_dict['policy_loss'] = policy_loss

            loss += args.lambda_1 * policy_loss
        elif args.dpo_type == 'dpo_only_smung':
            log_dict['chosen reward'] = chosen_reward
            log_dict['reject reward'] = reject_reward
            log_dict['pos_loss'] = pos_loss
            log_dict['neg_loss'] = neg_loss
            policy_loss = - F.logsigmoid(args.beta * (chosen_reward - reject_reward - args.lambda_2 * pos_loss))
            log_dict['policy loss'] = policy_loss
            
            loss = args.lambda_1 * policy_loss + length_loss
            log_dict['lm loss'] = loss

        elif args.dpo_type == 'smung_new':
            log_dict['chosen_reward'] = chosen_reward
            log_dict['reject_reward'] = reject_reward
            log_dict['pos_loss'] = pos_loss
            log_dict['neg_loss'] = neg_loss

            policy_loss = - F.logsigmoid(args.beta * (chosen_reward - reject_reward))
            log_dict['policy_loss'] = policy_loss

            loss = loss + args.lambda_1 * (policy_loss + args.lambda_2 * (pos_loss + neg_loss))
    return loss, log_dict

def forward_step(data_iterator, model, dpo_model=None):
    """Forward step."""
    timers = get_timers()
    tokenizer = get_tokenizer()
    # import pdb;
    # pdb.set_trace()
    # Get the batch.
    timers('batch-generator', log_level=2).start()
    args = get_args()
    if args.has_attention_masking:
        tokens, loss_mask, lm_labels, padding_mask, index_mask= get_batch(
        data_iterator)
    else:
        index_mask = None
        tokens, loss_mask, lm_labels, padding_mask = get_batch(
        data_iterator)
    timers('batch-generator').stop()
    output_tensors = model(tokens, padding_mask, index_attention_mask=index_mask)

    
    # lm loss
    lm_labels_new = lm_labels.transpose(0,1).contiguous()
    output_tensors_new = output_tensors[0].transpose(0,1).contiguous()
    if args.fp16_lm_cross_entropy:
        # print("fp16_loss")
        assert output_tensors_new.dtype == torch.half
        lm_loss = tensor_parallel.vocab_parallel_cross_entropy(output_tensors_new, lm_labels_new)
    else:
        lm_loss = tensor_parallel.vocab_parallel_cross_entropy(output_tensors_new.float(),
                                                        lm_labels_new)
        # [s, b] => [b s]
    lm_loss = lm_loss.transpose(0,1).contiguous()

    # length loss
    length_loss = None
    if args.length_predict:
        # print("length_loss")
        src_mask = index_mask[:,:1:,].squeeze(1)
        length_out = model.forward_length(output_tensors[1], src_mask)
        length_tgt = padding_mask.sum(1) - src_mask.sum(1)
        length_tgt = model.forward_length_prediction(length_out, length_tgt)
        length_logits = F.log_softmax(length_out, dim=-1)
        length_loss = F.nll_loss(length_logits, length_tgt, reduction="none").float().mean()
        length_loss = args.length_factor * length_loss
    
    # dpo loss
    pos_loss = None
    neg_loss = None
    chosen_reward = None
    reject_reward = None
    if args.dpo_training:
        if args.dpo_sampling_type == 'score_based':
            with torch.no_grad():
                dpo_output_tensors = dpo_model(tokens, padding_mask, index_attention_mask=index_mask)
                output_scores, output_tokens = F.softmax(dpo_output_tensors[0], -1).max(-1)
                
                lose_output_scores = output_scores.masked_fill(~tokens.eq(tokenizer.mask), 1.0)
                lose_scores_index = lose_output_scores.sort(-1)[1]
                win_output_scores = - output_scores.masked_fill(~tokens.eq(tokenizer.mask), 0.0)
                win_scores_index = win_output_scores.sort(-1)[1]
                
                len_ratio = random.uniform(0.2, 0.8)
                boundary_len = (loss_mask.bool().sum(1, keepdim=True).type_as(output_scores) * len_ratio).long()
                skeptical_mask = new_arange(loss_mask.bool()) < boundary_len
                win_skeptical_mask = skeptical_mask.clone().scatter(1, win_scores_index, skeptical_mask)
                lose_skeptical_mask = skeptical_mask.clone().scatter(1, lose_scores_index, skeptical_mask)

                win_prev_output_tokens = tokens.clone().masked_scatter_(win_skeptical_mask, output_tokens[win_skeptical_mask])
                lose_prev_output_tokens = tokens.clone().masked_scatter_(lose_skeptical_mask, output_tokens[lose_skeptical_mask])
                
                ref_win_outputs = dpo_model(win_prev_output_tokens, padding_mask, index_attention_mask=index_mask)
                ref_lose_outputs = dpo_model(lose_prev_output_tokens, padding_mask, index_attention_mask=index_mask)
                
                win_tokens = tokens.clone().masked_scatter_(win_skeptical_mask, output_tokens[win_skeptical_mask])
                ow_mask = win_prev_output_tokens.eq(tokenizer.mask)
                win_tokens = win_tokens.masked_scatter_(ow_mask, F.softmax(ref_win_outputs[0], -1).max(-1)[1][ow_mask])
                
                lose_tokens = tokens.clone().masked_scatter_(lose_skeptical_mask, output_tokens[lose_skeptical_mask])
                ol_mask = lose_prev_output_tokens.eq(tokenizer.mask)
                lose_tokens = lose_tokens.masked_scatter_(ol_mask, F.softmax(ref_lose_outputs[0], -1).max(-1)[1][ol_mask])

        elif args.dpo_sampling_type == 'random':
            with torch.no_grad(): 
                dpo_output_tensors = dpo_model(tokens, padding_mask, index_attention_mask=index_mask)
                output_scores, output_tokens = F.softmax(dpo_output_tensors[0], -1).max(-1)
                
                target_masks = loss_mask.bool().clone()
                target_score1 = output_tokens.clone().float().uniform_()
                target_score2 = output_tokens.clone().float().uniform_()
                target_score1.masked_fill_(~target_masks, 2.0)
                target_score2.masked_fill_(~target_masks, 2.0)
                target_index1 = target_score1.sort(1)[1]
                target_index2 = target_score2.sort(1)[1]

                len_ratio = random.uniform(0.2, 0.8)
                boundary_len = (loss_mask.bool().sum(1, keepdim=True).type_as(output_scores) * len_ratio).long()
                boundary_len = torch.clamp(boundary_len, min=1)
                skeptical_mask = new_arange(target_masks) < boundary_len
                
                skeptical_mask1 = skeptical_mask.clone().scatter(1, target_index1, skeptical_mask)
                skeptical_mask2 = skeptical_mask.clone().scatter(1, target_index2, skeptical_mask)

                prev_output_tokens1 = tokens.clone().masked_scatter_(skeptical_mask1, output_tokens[skeptical_mask1])
                prev_output_tokens2 = tokens.clone().masked_scatter_(skeptical_mask2, output_tokens[skeptical_mask2])
                
                ref_outputs1 = dpo_model(prev_output_tokens1, padding_mask, index_attention_mask=index_mask)
                ref_outputs2 = dpo_model(prev_output_tokens2, padding_mask, index_attention_mask=index_mask)
                
                out_tokens1 = tokens.clone().masked_scatter_(skeptical_mask1, output_tokens[skeptical_mask1])
                out_mask1 = prev_output_tokens1.eq(tokenizer.mask)
                out_tokens1 = out_tokens1.masked_scatter_(out_mask1, F.softmax(ref_outputs1[0], -1).max(-1)[1][out_mask1])
                
                out_tokens2 = tokens.clone().masked_scatter_(skeptical_mask2, output_tokens[skeptical_mask2])
                out_mask2 = prev_output_tokens2.eq(tokenizer.mask)
                out_tokens2 = out_tokens2.masked_scatter_(out_mask2, F.softmax(ref_outputs2[0], -1).max(-1)[1][out_mask2])

        win_policy_good = F.log_softmax(output_tensors[0], dim=-1).gather(dim=-1, index=win_tokens.unsqueeze(-1))
        win_policy_good_m = win_policy_good * loss_mask.bool().unsqueeze(-1)
        win_ref_good = F.log_softmax(dpo_output_tensors[0], dim=-1).gather(dim=-1, index=win_tokens.unsqueeze(-1))
        win_ref_good_m = win_ref_good * loss_mask.bool().unsqueeze(-1)
        win_policy_bad = F.log_softmax(output_tensors[0], dim=-1).gather(dim=-1, index=lose_tokens.unsqueeze(-1))
        win_policy_bad_m = win_policy_bad * loss_mask.bool().unsqueeze(-1)
        win_ref_bad = F.log_softmax(dpo_output_tensors[0], dim=-1).gather(dim=-1, index=lose_tokens.unsqueeze(-1))
        win_ref_bad_m = win_ref_bad * loss_mask.bool().unsqueeze(-1)

        chosen_reward = (win_policy_good_m - win_ref_good_m)[loss_mask.bool().unsqueeze(-1)].mean()
        reject_reward = (win_policy_bad_m - win_ref_bad_m)[loss_mask.bool().unsqueeze(-1)].mean()

        pos_loss = torch.clamp((win_ref_good_m - win_policy_good_m)[loss_mask.bool().unsqueeze(-1)].mean(), min=0.0)
        neg_loss = torch.clamp((win_ref_bad_m - win_policy_bad_m)[loss_mask.bool().unsqueeze(-1)].mean(), min=0.0)

    return lm_loss, partial(loss_func, loss_mask, length_loss, chosen_reward, reject_reward, pos_loss, neg_loss)

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GeBERT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        data_impl=args.data_impl,
        train_samples=train_val_test_num_samples[0],
        valid_samples=train_val_test_num_samples[1],
        max_seq_length=args.seq_length,
        has_sentence_split=args.has_sentence_split,
        masked_type=args.masked_type,
        masked_x_type=args.masked_x_type,
        has_attention_masking=args.has_attention_masking,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0("> finished creating GeBERT datasets ...")

    return train_ds, valid_ds, test_ds

if __name__ == "__main__":
    pretrain(train_valid_test_datasets_provider, model_provider,
             ModelType.encoder_or_decoder,
             forward_step, extra_args_provider=args_provider)
