

"""Pretrain GeBERT for generation tasks"""
from deepspeed.accelerator import get_accelerator
from megatron.initialize import initialize_megatron
from functools import partial
import jsonlines
import json_lines
import torch
import torch.nn.functional as F
from megatron.checkpointing import load_checkpoint
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.gebert_utils import build_train_valid_test_datasets
from megatron.model import GeBertModel
from megatron.utils import average_losses_across_data_parallel_group
import argparse
from megatron.arguments import core_transformer_config_from_args
from megatron.training import get_model
from megatron import get_tokenizer
import numpy as np
from fairseq.utils import new_arange
from megatron.initialize import set_jit_fusion_options
import deepspeed
from tqdm import tqdm

def args_provider(parser):
    group = parser.add_argument_group(title='Extra args')

    # data args
    group.add_argument('--inputfile', type=str, default='', help='input file')
    group.add_argument('--outfile', type=str, default='', help='output file.')
    group.add_argument('--extra-outfile', type=str, default=None, help='extra output file.')

    # length prediction args
    group.add_argument('--length-predict', action='store_true', help='if we adopt length prediction')
    group.add_argument('--max-predict-length', type=int, help='the maximun predicted length')
    group.add_argument('--length-factor', type=float, default=0.1, help='length loss factor')
    group.add_argument('--load-LP-module', action='store_true', help='if we load length prediction')

    # inferene args 
    group.add_argument('--inf-type',type=str, default='diff_gram_based', help='')
    group.add_argument('--max-iter',type=int, default=1)
    group.add_argument('--length-beam', type=int, default=1, help='The length beam.')
    group.add_argument('--position-beam', type=int, default=1, help='The position beam number.')
    group.add_argument('--tokens-beam', type=int, default=1, help='The tokens beam number.')
    group.add_argument('--has-attention-masking', action='store_true', help='If adopt the attention masking.')
    return parser


def model_provider(pre_process=True, post_process=False):
    """Build the model."""

    print_rank_0('building GEBERT model ...')
    args = get_args()
    config = core_transformer_config_from_args(args)
    model = GeBertModel(
            config=config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=True)
    return model


def inferece(model, batch_sample, tokenizer, max_seq_length, max_iter, device, length_beam=1, inference_type='direct', beam_position=1, beam_tokens=1):
    source_list = []
    target_list = []
    index_mask_source = []
    for single_sample in batch_sample:
        source = single_sample["source"]
        tokenized_ids_source = tokenizer.tokenize(source)[:max_seq_length-1]
        source_list.append(tokenized_ids_source)
        target = single_sample["target"]
        tokenized_ids_target = tokenizer.tokenize(target)[:max_seq_length-1]
        target_list.append(tokenized_ids_target)
        attention_index = len(tokenized_ids_source) + 1

        index_mask=None
        if args.has_attention_masking:
            index_mask = np.zeros((max_seq_length, max_seq_length))
            index_mask[:attention_index, :attention_index]=1
            index_mask = torch.tensor(index_mask)
            index_mask = index_mask.unsqueeze(0).to(device)
            index_mask_source.append(index_mask)


    batch_index_mask_source = torch.cat(index_mask_source, dim=0)
    max_length_source = max_seq_length -1 
    source_list = [[tokenizer.bos] + i + [tokenizer.pad for _ in range(max_length_source-len(i))] for i in source_list]
    batch_source = torch.tensor(source_list).to(device)
    source_padding_mask = batch_source.ne(tokenizer.pad).type_as(batch_source)
    output_tensors = model(batch_source, source_padding_mask, batch_index_mask_source)
    
    #length——prediction
    length_out = model.forward_length(output_tensors[1], source_padding_mask).detach()
    length_tgt = length_out.topk(length_beam, dim=-1, largest=True, sorted=True)[1]
    
    token_list = []
    index_mask_list = []
    attention_index_list = []

    for single_sample_index in range(len(batch_sample)):
        single_sample = batch_sample[single_sample_index]
        source = single_sample["source"]
        tokenized_ids_source = tokenizer.tokenize(source)

        for beam_length_index in range(length_tgt.size(1)):
            length_target = length_tgt[single_sample_index][beam_length_index].tolist()
            sample_len = len(tokenized_ids_source) + length_target
            num_tokens = sample_len+3
            if num_tokens > max_seq_length:
                trun_index = max_seq_length - length_target - 3
                tokenized_ids_source_new = tokenized_ids_source[:trun_index] 
                num_tokens = max_seq_length
            else:
                tokenized_ids_source_new = tokenized_ids_source

            prev_tokenized_ids_target = np.full(length_target, tokenizer.mask).tolist()
            pad_list = [tokenizer.pad for _ in range(max_seq_length-num_tokens)]
            input_ids = [[tokenizer.bos] + tokenized_ids_source_new + [tokenizer.bos] + prev_tokenized_ids_target + [tokenizer.eos] + pad_list]
            assert len(input_ids[0])==max_seq_length
            tokens = torch.tensor(input_ids).to(device)
            padding_mask = tokens.ne(tokenizer.pad).type_as(tokens)
            attention_index = len(tokenized_ids_source_new) + 1
            attention_index_list.append(attention_index)

            index_mask=None
            if args.has_attention_masking:
                index_mask = np.zeros((max_seq_length, max_seq_length))
                index_mask[:attention_index, :attention_index]=1
                index_mask[attention_index:num_tokens, :num_tokens]=1
                index_mask = torch.tensor(index_mask)
                index_mask = index_mask.unsqueeze(0).to(device)
        
            token_list.append(tokens)
            index_mask_list.append(index_mask)
    
    
    if inference_type=='direct':
        batch_tokens = torch.cat(token_list,dim=0)
        batch_index_mask = torch.cat(index_mask_list, dim=0)
        decoder_option = {
            "current_step": 0,
            "max_step": max_iter,
            "current_ids": batch_tokens.clone(),
            "current_confidence": None,
        }

        for step in range(max_iter):
            decoder_option["current_step"] = step
            decoder_option = forward_target(model=model, tokenizer=tokenizer, decoder_option=decoder_option, padding_mask=padding_mask, index_mask=batch_index_mask)

        result_avg = []
        result = []
        for single_batch_sample_index in range(0,len(token_list),length_beam):
            single_batch_mask_number = batch_tokens[single_batch_sample_index:single_batch_sample_index+length_beam].eq(tokenizer.mask).sum(1)
            single_batch_sample_ids = decoder_option["current_ids"][single_batch_sample_index:single_batch_sample_index+length_beam]
            single_batch_sample_confidence = decoder_option["current_confidence"][single_batch_sample_index:single_batch_sample_index+length_beam]

            single_batch_sample_max_index = single_batch_sample_confidence.sum(1).max(-1)[1]
            single_batch_sample_attention_index = attention_index_list[single_batch_sample_index:single_batch_sample_index+length_beam]
            result.append(tokenizer.detokenize(single_batch_sample_ids[single_batch_sample_max_index][single_batch_sample_attention_index[single_batch_sample_max_index]:].tolist(), skip_special_tokens=True))
            
            single_batch_sample_max_index_avg = (single_batch_sample_confidence.sum(1)/single_batch_mask_number).max(-1)[1]
            single_batch_sample_attention_index = attention_index_list[single_batch_sample_index:single_batch_sample_index+length_beam]
            result_avg.append(tokenizer.detokenize(single_batch_sample_ids[single_batch_sample_max_index_avg][single_batch_sample_attention_index[single_batch_sample_max_index_avg]:].tolist(), skip_special_tokens=True))

        return result, result_avg

    elif inference_type=='position_beam_simple':
        final_result = []
        extra_final_result = []
        score_list = []
        avg_score_list = []
        ids_list = []
        confidence_list = []
        beam = beam_position

        for sample_index in range(len(token_list)):
            batch_tokens = torch.cat([token_list[sample_index]],dim=0)
            batch_index_mask = torch.cat([index_mask_list[sample_index]], dim=0)

            decoder_option = {
                "current_step": 0,
                "max_step": max_iter,
                "current_ids": batch_tokens.clone(),
                "current_confidence": None,
            }
            
            for step in range(max_iter):
                decoder_option["current_step"] = step
                decoder_option = forward_target_beam_position_simple(model=model, tokenizer=tokenizer, decoder_option=decoder_option, padding_mask=None, index_mask=batch_index_mask, beam=beam)

            confidence_list.append(decoder_option['current_confidence'])
            ids_list.append(decoder_option['current_ids'])

        single_batch_mask_number = torch.cat(token_list,dim=0).eq(tokenizer.mask).sum(1)
        score_list = [i.sum(1).unsqueeze(0) for i in confidence_list]
        avg_score_list =torch.cat(score_list,dim=0)/single_batch_mask_number.unsqueeze(-1)

        max_length_index = torch.cat(score_list,dim=0).max(-1)[0].max(-1)[1]
        max_beam_index = torch.cat(score_list,dim=0)[max_length_index].max(-1)[1]
        final_result.append(tokenizer.detokenize((ids_list[max_length_index][max_beam_index].tolist())[attention_index_list[max_length_index]:], skip_special_tokens=True))
        
        extra_max_length_index = avg_score_list.max(-1)[0].max(-1)[1]
        extra_max_beam_index = avg_score_list[extra_max_length_index].max(-1)[1]
        extra_final_result.append(tokenizer.detokenize((ids_list[extra_max_length_index][extra_max_beam_index].tolist())[attention_index_list[max_length_index]:],skip_special_tokens=True))
        
        return final_result, extra_final_result

    elif inference_type=='tokens_beam':
        extra_final_result = []
        final_result = []
        beam = beam_tokens
        score_list = []
        ids_list = []
        confidence_list = []
        avg_score_list = []
        for sample_index in range(len(token_list)):
            batch_tokens = torch.cat([token_list[sample_index]],dim=0)
            batch_index_mask = torch.cat([index_mask_list[sample_index]], dim=0)
            
            decoder_option = {
                "current_step": 0,
                "max_step": max_iter,
                "current_ids": batch_tokens.clone(),
                "current_confidence": None,
            }

            for step in range(max_iter):
                decoder_option["current_step"] = step
                decoder_option = forward_target_beam_tokens(model=model, tokenizer=tokenizer, decoder_option=decoder_option, padding_mask=padding_mask, index_mask=batch_index_mask, beam=beam)

            confidence_list.append(decoder_option['current_confidence'])
            ids_list.append(decoder_option['current_ids'])
        
        single_batch_mask_number = torch.cat(token_list,dim=0).eq(tokenizer.mask).sum(1)

        score_list = [i.sum(1).tolist() for i in confidence_list]
        avg_score_list =(torch.tensor(score_list).type_as(decoder_option['current_confidence'])/single_batch_mask_number.unsqueeze(-1)).tolist()

        max_length_index = torch.tensor(score_list).max(-1)[0].max(-1)[1].tolist()
        max_beam_index = torch.tensor(score_list[max_length_index]).max(-1)[1].tolist()
        final_result.append(tokenizer.detokenize((ids_list[max_length_index][max_beam_index].tolist())[attention_index_list[0]:], skip_special_tokens=True))
        
        extra_max_length_index = torch.tensor(avg_score_list).max(-1)[0].max(-1)[1].tolist()
        extra_max_beam_index = torch.tensor(score_list[extra_max_length_index]).max(-1)[1].tolist()
        extra_final_result.append(tokenizer.detokenize((ids_list[extra_max_length_index][extra_max_beam_index].tolist())[attention_index_list[0]:],skip_special_tokens=True))
        return final_result, extra_final_result
    
def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (
        (output_masks.sum(1, keepdim=True).type_as(output_scores)) * p
    ).long()
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)

def nucleus_sampling(probs):
        
        nucleus_p = 0.9
        nucleus_k = 100
        temperature = 1.0
        probs = F.softmax(probs / temperature, dim=-1)
        raw_indices_buf = probs.max(-1)[1].unsqueeze(-1)
        
        if nucleus_p > 0:
            sorted_probs, sorted_indices = probs.sort(descending=True)
            cumsum_probs = sorted_probs.cumsum(dim=2)
            mask = cumsum_probs.lt(nucleus_p)

            cumsum_mask = mask.cumsum(dim=2)
            last_included = cumsum_mask[:, :, -1:]
            last_included.clamp_(0, mask.size()[2] - 1)
            mask = mask.scatter_(2, last_included, 1)
            
            max_dim = last_included.max()
            truncated_mask = mask[:, :, : max_dim + 1]
            truncated_probs = sorted_probs[:, :, : max_dim + 1]
            truncated_indices = sorted_indices[:, :, : max_dim + 1]
            trimed_probs = truncated_probs.masked_fill_(~truncated_mask, 0)
        else:
            trimed_probs, truncated_indices = probs.topk(nucleus_k)
        
        bsz, seq_len, _ = trimed_probs.size()
        select_buf = torch.multinomial(trimed_probs.view(bsz * seq_len, -1), 1, replacement=True).view(bsz, seq_len)
        scores_buf = torch.gather(trimed_probs, dim=2, index=select_buf.unsqueeze(-1))
        indices_buf = torch.gather(truncated_indices, dim=2, index=select_buf.unsqueeze(-1))
        
        return torch.log(scores_buf).squeeze(-1), indices_buf.squeeze(-1)

def forward_target_beam_position(model, tokenizer, decoder_option, padding_mask, index_mask, beam):
    current_step = decoder_option["current_step"]
    max_step = decoder_option["max_step"]
    output_ids = decoder_option["current_ids"]
    output_confidence = decoder_option["current_confidence"]
    current_mask_index = output_ids.eq(tokenizer.mask)
    output_tensors = model(output_ids, padding_mask, index_mask)[0].detach()

    prediction_confidence, prediction_ids = F.log_softmax(output_tensors, -1).max(-1)
    if current_step==0:
        output_confidence = output_ids.ne(tokenizer.mask).type_as(prediction_confidence)
        decoder_option["mask_able_index"] = current_mask_index
    output_ids.masked_scatter_(current_mask_index, prediction_ids[current_mask_index])
    output_confidence.masked_scatter_(current_mask_index, prediction_confidence[current_mask_index])

    if (current_step + 1) < max_step:
        next_predict_ids_list = [] 
        next_predict_confidence_list = []
        next_sum_score_list = []
        mask_able_index = decoder_option["mask_able_index"][0]
        length_unmasking = (mask_able_index.sum(0) - (mask_able_index.sum(0) * (1 - (current_step + 1) / max_step)).long())
        next_prediction_ids_confidence_dict = {}
        for beam_sample_index in range(output_ids.size(0)):
            
            beam_sample_confidence = output_confidence[beam_sample_index][mask_able_index].tolist()
            top_k_index_list = find_max_sum_combinations(beam_sample_confidence, length_unmasking.tolist(), beam)
            for sum_score, index_list in top_k_index_list:
                if index_list not in next_prediction_ids_confidence_dict.keys():
                    next_prediction_ids_confidence_dict[index_list] = (sum_score, beam_sample_index)
                else:
                    if next_prediction_ids_confidence_dict[index_list][0] < sum_score:
                        next_prediction_ids_confidence_dict[index_list] = (sum_score, beam_sample_index)
        
        next_prediction_ids_confidence_dict = {k: v for k, v in sorted(next_prediction_ids_confidence_dict.items(), key=lambda item: item[1][0], reverse=True)[:beam]}
        

        for final_index_list, sum_score in next_prediction_ids_confidence_dict.items():
            beam_sample_ids = output_ids[sum_score[1]][mask_able_index].tolist()
            single_mask_ids = np.full(len(beam_sample_ids), tokenizer.mask).tolist()
            for j_index in final_index_list:
                single_mask_ids[j_index] = beam_sample_ids[j_index]
            
            next_predict_ids_list.append(output_ids[sum_score[1]].clone().masked_scatter_(mask_able_index,torch.tensor(single_mask_ids).type_as(output_ids)).tolist())
            next_predict_confidence_list.append(output_confidence[sum_score[1]].tolist())
            next_sum_score_list.append(sum_score[0])

        decoder_option['current_ids'] = torch.tensor(next_predict_ids_list).type_as(output_ids)
        decoder_option['current_confidence'] = torch.tensor(next_predict_confidence_list).type_as(output_confidence)
        decoder_option['sum_score'] = next_sum_score_list

    return decoder_option

def forward_target_beam_position_simple(model, tokenizer, decoder_option, padding_mask, index_mask, beam):
    current_step = decoder_option["current_step"]
    max_step = decoder_option["max_step"]
    output_ids = decoder_option["current_ids"]
    output_confidence = decoder_option["current_confidence"]
    current_mask_index = output_ids.eq(tokenizer.mask)
    output_tensors = model(output_ids, padding_mask, index_mask)[0].detach()
    prediction_confidence, prediction_ids = F.log_softmax(output_tensors, -1).max(-1)
    if current_step==0:
        output_confidence = torch.zeros_like(prediction_confidence).type_as(prediction_confidence)
        decoder_option["mask_able_index"] = current_mask_index

    decoder_option["mask_able_index"] = decoder_option["mask_able_index"][0].repeat(output_ids.size(0),1)
    output_ids.masked_scatter_(current_mask_index, prediction_ids[current_mask_index])
    output_confidence.masked_scatter_(current_mask_index, prediction_confidence[current_mask_index])
    
    if (current_step + 1) < max_step:
        next_predict_ids_list = [] 
        next_predict_confidence_list = []
        next_sum_score_list = []

        sorted_index = output_confidence.sort(-1)[1]
        boundary_len = (
            (decoder_option["mask_able_index"].sum(1, keepdim=True).type_as(output_confidence)) * (1 - (current_step + 1) / max_step)
        ).long()
        skeptical_mask = new_arange(decoder_option["mask_able_index"]) < boundary_len
        skeptical_mask = skeptical_mask.scatter(1, sorted_index, skeptical_mask)

        output_ids.masked_fill_(skeptical_mask,tokenizer.mask)
        output_confidence.masked_fill_(skeptical_mask, 0.0)

        length_unmasking = decoder_option["mask_able_index"].sum(1) - skeptical_mask.sum(1)
        
        next_prediction_ids_confidence_dict = {}

        for beam_sample_index in range(output_ids.size(0)):
            beam_sample_ids = output_ids[beam_sample_index]
            beam_sample_confidence = output_confidence[beam_sample_index]
            beam_length_unmasking = length_unmasking.tolist()[beam_sample_index]
            beam_bounary_len = boundary_len.tolist()[beam_sample_index][0]
            beam_sorted_index = sorted_index[beam_sample_index]
            next_prediction_ids_confidence_dict[(beam_sample_ids,beam_sample_confidence)] = sum(beam_sample_confidence.tolist())       
            current_sample_number = 1 
            able_beam_number = min(beam_length_unmasking, beam_bounary_len)
            for candidate_position_index in range(able_beam_number):
                if current_sample_number >= beam:
                    continue
                new_prediction_ids = beam_sample_ids.clone()
                new_prediction_confidence = beam_sample_confidence.clone()
                current_replace_index = beam_sorted_index[beam_bounary_len+candidate_position_index]
                current_remove_index = beam_sorted_index[beam_bounary_len-candidate_position_index]
                new_prediction_ids[current_replace_index] = prediction_ids[beam_sample_index][current_replace_index]
                new_prediction_confidence[current_replace_index] = prediction_confidence[beam_sample_index][current_replace_index]
                new_prediction_ids[current_remove_index] = tokenizer.mask
                new_prediction_confidence[current_remove_index] = 0.0
                
                if (new_prediction_ids,new_prediction_confidence) not in next_prediction_ids_confidence_dict.keys():
                    next_prediction_ids_confidence_dict[(new_prediction_ids,new_prediction_confidence)] = sum(new_prediction_confidence.tolist())
                    current_sample_number+=1
                else:
                    if next_prediction_ids_confidence_dict[(new_prediction_ids,new_prediction_confidence)] < sum(new_prediction_confidence.tolist()):
                        next_prediction_ids_confidence_dict[(new_prediction_ids,new_prediction_confidence)] = sum(new_prediction_confidence.tolist())
                        current_sample_number+=1


        next_prediction_ids_confidence_dict = {k: v for k, v in sorted(next_prediction_ids_confidence_dict.items(), key=lambda item: item[1], reverse=True)[:beam]}
        
        
        for final_prediction_ids, final_prediction_confidence in next_prediction_ids_confidence_dict.keys():
            next_predict_ids_list.append(final_prediction_ids.tolist())
            next_predict_confidence_list.append(final_prediction_confidence.tolist())
            next_sum_score_list.append(next_prediction_ids_confidence_dict[(final_prediction_ids, final_prediction_confidence)])

        decoder_option['current_ids'] = torch.tensor(next_predict_ids_list).type_as(output_ids)
        decoder_option['current_confidence'] = torch.tensor(next_predict_confidence_list).type_as(output_confidence)
        decoder_option['sum_score'] = next_sum_score_list
    
    return decoder_option

def forward_target(model, tokenizer, decoder_option, padding_mask, index_mask):
    current_step = decoder_option["current_step"]
    max_step = decoder_option["max_step"]
    output_ids = decoder_option["current_ids"]
    output_confidence = decoder_option["current_confidence"]
    current_mask_index = output_ids.eq(tokenizer.mask)
    output_tensors = model(output_ids, padding_mask, index_mask)[0].detach()
    prediction_confidence, prediction_ids = F.log_softmax(output_tensors, -1).max(-1)
    if current_step==0:
        output_confidence = torch.zeros_like(prediction_confidence).type_as(prediction_confidence)
        decoder_option["mask_able_index"] = current_mask_index
    output_ids.masked_scatter_(current_mask_index, prediction_ids[current_mask_index])
    output_confidence.masked_scatter_(current_mask_index, prediction_confidence[current_mask_index])
    if (current_step + 1) < max_step:
        skeptical_mask = _skeptical_unmasking(
            output_confidence, decoder_option["mask_able_index"],  1 - (current_step + 1) / max_step, 
        )
        output_ids.masked_fill_(skeptical_mask,tokenizer.mask)
        output_confidence.masked_fill_(skeptical_mask, 0.0)
        decoder_option["current_ids"] = output_ids.detach()
        decoder_option["current_confidence"] = output_confidence.detach()
    return decoder_option

def forward_target_beam_tokens(model, tokenizer, decoder_option, padding_mask, index_mask, beam):
    current_step = decoder_option["current_step"]
    max_step = decoder_option["max_step"]
    output_ids = decoder_option["current_ids"]
    output_confidence = decoder_option["current_confidence"]
    current_mask_index = output_ids.eq(tokenizer.mask)
    output_tensors = model(output_ids, padding_mask, index_mask)[0].detach()
    
    top_k_confidence, top_k_ids = F.log_softmax(output_tensors, -1).topk(beam, dim=-1, largest=True, sorted=True)
    top_0_confidence = top_k_confidence[:,:,0]
    top_0_ids = top_k_ids[:,:,0]

    if current_step==0:
        output_confidence = torch.zeros_like(top_0_confidence).type_as(top_0_confidence)
        decoder_option["mask_able_index"] = current_mask_index

    decoder_option["mask_able_index"] = decoder_option["mask_able_index"][0].repeat(output_ids.size(0),1)
    output_ids.masked_scatter_(current_mask_index, top_0_ids[current_mask_index])
    output_confidence.masked_scatter_(current_mask_index, top_0_confidence[current_mask_index])

    if (current_step + 1) < max_step:
        
        next_predict_ids_list = []
        next_predict_confidence_list = []
        next_sum_score_list = []

        sorted_index = output_confidence.sort(-1)[1]
        boundary_len = (
            (decoder_option["mask_able_index"].sum(1, keepdim=True).type_as(output_confidence) - 2 ) * (1 - (current_step + 1) / max_step)
        ).long()
        skeptical_mask = new_arange(decoder_option["mask_able_index"]) < boundary_len
        skeptical_mask = skeptical_mask.scatter(1, sorted_index, skeptical_mask)

        output_ids.masked_fill_(skeptical_mask,tokenizer.mask)
        output_confidence.masked_fill_(skeptical_mask, 0.0)
        
        length_unmasking = decoder_option["mask_able_index"].sum(1) - skeptical_mask.sum(1)

        next_prediction_ids_confidence_dict = {}
             
        for beam_sample_index in range(output_ids.size(0)):
            beam_sample_ids = output_ids[beam_sample_index]
            beam_sample_confidence = output_confidence[beam_sample_index]
            beam_length_unmasking = length_unmasking.tolist()[beam_sample_index]
            beam_bounary_len = boundary_len.tolist()[beam_sample_index][0]
            beam_sorted_index = sorted_index[beam_sample_index]
            next_prediction_ids_confidence_dict[(beam_sample_ids,beam_sample_confidence)] = sum(beam_sample_confidence.tolist()) 
            candidate_prediction_index = 1       
            candidate_token_index = 0
            current_sample_number = 1 
            while current_sample_number < beam:
                # import pdb; pdb.set_trace()
                new_prediction_ids = beam_sample_ids.clone()
                new_prediction_confidence = beam_sample_confidence.clone()
                current_replace_index = beam_sorted_index[beam_bounary_len+candidate_token_index]
                new_prediction_ids[current_replace_index] = top_k_ids[:,:,candidate_prediction_index][0][current_replace_index]
                new_prediction_confidence[current_replace_index] = top_k_confidence[:,:,candidate_prediction_index][0][current_replace_index]
                if (new_prediction_ids,new_prediction_confidence) not in next_prediction_ids_confidence_dict.keys():
                    next_prediction_ids_confidence_dict[(new_prediction_ids,new_prediction_confidence)] = sum(new_prediction_confidence.tolist())
                    current_sample_number+=1
                else:
                    if next_prediction_ids_confidence_dict[(new_prediction_ids,new_prediction_confidence)] < sum(new_prediction_confidence.tolist()):
                        next_prediction_ids_confidence_dict[(new_prediction_ids,new_prediction_confidence)] = sum(new_prediction_confidence.tolist())
                        current_sample_number+=1

                if candidate_token_index + 1 < beam_length_unmasking:
                    candidate_token_index+=1
                else:
                    candidate_token_index=0
                    candidate_prediction_index+=1

        next_prediction_ids_confidence_dict = {k: v for k, v in sorted(next_prediction_ids_confidence_dict.items(), key=lambda item: item[1], reverse=True)[:beam]}
        
        
        for final_prediction_ids, final_prediction_confidence in next_prediction_ids_confidence_dict.keys():
            next_predict_ids_list.append(final_prediction_ids.tolist())
            next_predict_confidence_list.append(final_prediction_confidence.tolist())
            next_sum_score_list.append(next_prediction_ids_confidence_dict[(final_prediction_ids, final_prediction_confidence)])

        decoder_option['current_ids'] = torch.tensor(next_predict_ids_list).type_as(output_ids)
        decoder_option['current_confidence'] = torch.tensor(next_predict_confidence_list).type_as(output_confidence)
        decoder_option['sum_score'] = next_sum_score_list

    return decoder_option

if __name__ == "__main__":
    ds_dict = {
        "train_batch_size" : 1,
        "train_micro_batch_size_per_gpu": 1,
        "steps_per_print": 10,

        "zero_optimization": {
        "stage": 0
        },

        "gradient_clipping": 1.0,
        "prescale_gradients": False,

        "fp16": {
        "enabled": False,
        },

        "bf16": {
        "enabled": True,
        },

        "wall_clock_breakdown" : False
        }

    device = torch.device("cuda")
    initialize_megatron(extra_args_provider=args_provider,args_defaults={'tokenizer_type': 'HFTokenizer'})
    args = get_args()
    tokenizer = get_tokenizer()
    model = get_model(model_provider)
    if args.deepspeed:
        if get_accelerator().device_name() == 'cuda':
            set_jit_fusion_options()
        args.deepspeed_config_dict = ds_dict
        model, optimizer, _, opt_param_scheduler = deepspeed.initialize(
                    model=model[0],
                    optimizer=None,
                    args=args,
                    lr_scheduler=None,
                    mpu=None,
                    config=args.deepspeed_config_dict,
                )
        _ = load_checkpoint([model], None, None, strict=True, load_only_weights=False)
    else:
        _ = load_checkpoint(model, None, None, strict=False, load_only_weights=False)
        assert len(model) == 1, "Above condition should have caught this"
        model = model[0].to(device)
    # assert len(model) == 1, "Above condition should have caught this"
    
    model.eval()
    
    with open(args.inputfile, 'rb') as f: 
        item_list = [item for item in json_lines.reader(f)]

    max_seq_length = args.seq_length
    max_iter = args.max_iter
    batch_size = args.micro_batch_size 
    inference_type = args.inftype
    length_beam = args.length_beam
    beam_position = args.position_beam
    beam_tokens = args.tokens_beam

    for index in tqdm(range(0,len(item_list),batch_size)):
        batch_sample = item_list[index:index+batch_size]
        final_output_tokens, extra__output_tokens = inferece(model, batch_sample, tokenizer, max_seq_length, max_iter, device, length_beam, inference_type, beam_position, beam_tokens)
        
        for single_output in final_output_tokens:
            with open(args.outfile, "a", encoding="utf-8") as file:
                file.write(single_output + "\n")

        if args.extra_outfile is not None:
            for single_output in extra__output_tokens:
                with open(args.extra_outfile, "a", encoding="utf-8") as file:
                    file.write(single_output + "\n") 



