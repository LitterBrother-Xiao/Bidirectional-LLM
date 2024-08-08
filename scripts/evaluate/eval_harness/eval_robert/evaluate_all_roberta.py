from functools import reduce
from logging import logMultiprocessing
import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
#                                              os.path.pardir,os.path.pardir)))
from fairseq.utils import new_arange
from lm_eval.api.model import LM
from lm_eval import evaluator, tasks, utils
from lm_eval.api.model import CacheHook
from tqdm import tqdm
import torch.nn.functional as F
import math
# from lm_eval.tasks import ALL_TASKS
# from pretrain_gpt import model_provider
import numpy as np
import time
import torch

import pickle
import json


class EvalHarnessAdaptor(LM):
    def __init__(self, args, model, tokenizer, device):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.VOCAB_SIZE = tokenizer.vocab_size
        self.mask_token_id = tokenizer.mask_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.inftype = args.inftype
        self._max_length = args.seq_length
        self._max_iter = args.max_iter
        self._rank = 0
        self._world_size = 1
        self._device = device

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device
    
    def loglikelihood_rolling(self, requests):
        import pdb; pdb.set_trace()
        return None
    
    def loglikelihood(self, requests):
        if self.inftype == "order_based":
            tokenizer = self.tokenizer
            model = self.model
            result = []
            device = self._device
            max_seq_length = self._max_length
            for context, continuation in tqdm([req.args for req in requests]):
                source = context
                target = continuation
                tokenized_ids_source = tokenizer.tokenize(source)
                tokenized_ids_all = tokenizer.tokenize(source+target)
                tokenized_ids_target = tokenized_ids_all[len(tokenized_ids_source):]
                sample_len = len(tokenized_ids_target) + len(tokenized_ids_source)
                num_tokens = sample_len + 1
                if num_tokens > max_seq_length:
                    trun_index = max_seq_length - len(tokenized_ids_target) - 1
                    tokenized_ids_source = tokenized_ids_source[max_seq_length-trun_index:] 
                    num_tokens = max_seq_length
                pad_list = [tokenizer.pad for _ in range(max_seq_length-num_tokens)]
                prev_tokenized_ids_target = np.full(len(tokenized_ids_target), tokenizer.mask_token_id).tolist()
                input_ids = [[tokenizer.bos] + tokenized_ids_source + prev_tokenized_ids_target + pad_list]
                assert len(input_ids[0])==max_seq_length
                tokens = torch.tensor(input_ids).to(device)
                padding_mask = tokens.ne(tokenizer.pad).type_as(tokens)
                index_mask = np.zeros((max_seq_length, max_seq_length))
                attention_index = len(tokenized_ids_source) + 1
                index_mask[:attention_index, :attention_index]=1
                index_mask[attention_index:num_tokens, :num_tokens]=1
                index_mask = torch.tensor(index_mask)
                index_mask = index_mask.unsqueeze(0).to(device)
                order_list = np.arange(len(tokenized_ids_target)).tolist()
                
                tokenized_ids_target = torch.tensor(tokenized_ids_target).to(device)
                current_mask = tokenized_ids_target.ne(tokenizer.mask_token_id)
                pro_list = []
                for j in range(len(order_list)):
                    current_tokenized_ids_target = tokenized_ids_target.clone().masked_fill_(current_mask,tokenizer.mask_token_id).tolist()
                    current_tokens_list = [[tokenizer.bos] + tokenized_ids_source + current_tokenized_ids_target + pad_list]
                    assert len(current_tokens_list[0])==max_seq_length
                    current_tokens = torch.tensor(current_tokens_list).to(device)
                    # import pdb; pdb.set_trace()
                    output_tensors = F.softmax(model(current_tokens, padding_mask, index_mask)[0].detach(), dim=-1)
                    current_scores_list = output_tensors[0][attention_index:attention_index+len(tokenized_ids_target)]
                    current_scores = torch.gather(current_scores_list,1,tokenized_ids_target.unsqueeze(-1))
                    pro_list.append(current_scores[order_list[j]].tolist())
                    current_mask[order_list[j]] = False
                
                result.append((np.log(np.prod(pro_list)), False))
                # import pdb; pdb.set_trace()
            return result

        if self.inftype == "gram_based":
            tokenizer = self.tokenizer
            model = self.model
            result = []
            device = self._device
            beam = 1
            max_seq_length = self._max_length
            for context, continuation in tqdm([req.args for req in requests]):
                source = context
                target = continuation
                tokenized_ids_source = tokenizer.tokenize(source)
                tokenized_ids_all = tokenizer.tokenize(source+target)
                # import pdb; pdb.set_trace()
                tokenized_ids_target = tokenized_ids_all[len(tokenized_ids_source):]
                sample_len = len(tokenized_ids_target) + len(tokenized_ids_source)
                num_tokens = sample_len + 1
                if num_tokens > max_seq_length:
                    trun_index = max_seq_length - len(tokenized_ids_target) - 1
                    tokenized_ids_source = tokenized_ids_source[max_seq_length-trun_index:] 
                    num_tokens = max_seq_length
                pad_list = [tokenizer.pad for _ in range(max_seq_length-num_tokens)]
                # prev_tokenized_ids_target = np.full(len(tokenized_ids_target), tokenizer.mask_token_id).tolist()
                input_ids = [[tokenizer.bos] + tokenized_ids_source + tokenized_ids_target + pad_list]
                tokenized_ids_target_ts = torch.tensor(tokenized_ids_target).to(device)
                assert len(input_ids[0])==max_seq_length
                tokens = torch.tensor(input_ids).to(device)
                padding_mask = tokens.ne(tokenizer.pad).type_as(tokens)
                index_mask = np.zeros((max_seq_length, max_seq_length))
                attention_index = len(tokenized_ids_source) + 1
                index_mask[:attention_index, :attention_index]=1
                index_mask[attention_index:num_tokens, :num_tokens]=1
                index_mask = torch.tensor(index_mask)
                index_mask = index_mask.unsqueeze(0).to(device)

                gram_list = [len(tokenized_ids_target)]
                for ngarm in gram_list:
                    pro_list1 = []
                    for j in range(len(tokenized_ids_target) - ngarm + 1):
                        current_tokens = tokens.clone()
                        current_tokens[0][attention_index+j:attention_index+j+ngarm] = tokenizer.mask
                        output_tensors = F.softmax(model(current_tokens, padding_mask, index_mask)[0].detach(), dim=-1)
                        score = []
                        for k in range(ngarm):
                            target_index = tokenized_ids_target[j+k]
                            score.append(output_tensors[0][attention_index+j+k][target_index].tolist())
                        pro_list1.append(np.prod(score))
                
                result.append((np.log(pro_list1[0]),False))
            
            return result
        
        if self.inftype == "min_beam_based":
            tokenizer = self.tokenizer
            model = self.model
            result = []
            device = self._device
            beam = 1
            max_seq_length = self._max_length
            for context, continuation in tqdm([req.args for req in requests]):
                source = context
                target = continuation
                tokenized_ids_source = tokenizer.tokenize(source)
                tokenized_ids_all = tokenizer.tokenize(source+target)
                tokenized_ids_target = tokenized_ids_all[len(tokenized_ids_source):]
                sample_len = len(tokenized_ids_target) + len(tokenized_ids_source)
                num_tokens = sample_len + 1
                if num_tokens > max_seq_length:
                    trun_index = max_seq_length - len(tokenized_ids_target) - 1
                    tokenized_ids_source = tokenized_ids_source[max_seq_length-trun_index:] 
                    num_tokens = max_seq_length
                pad_list = [tokenizer.pad for _ in range(max_seq_length-num_tokens)]
                prev_tokenized_ids_target = np.full(len(tokenized_ids_target), tokenizer.mask_token_id).tolist()
                input_ids = [[tokenizer.bos] + tokenized_ids_source + prev_tokenized_ids_target + pad_list]
                tokenized_ids_target_ts = torch.tensor(tokenized_ids_target).to(device)
                assert len(input_ids[0])==max_seq_length
                tokens = torch.tensor(input_ids).to(device)
                padding_mask = tokens.ne(tokenizer.pad).type_as(tokens)
                index_mask = np.zeros((max_seq_length, max_seq_length))
                attention_index = len(tokenized_ids_source) + 1
                index_mask[:attention_index, :attention_index]=1
                index_mask[attention_index:num_tokens, :num_tokens]=1
                index_mask = torch.tensor(index_mask)
                index_mask = index_mask.unsqueeze(0).to(device)
                token_mask = tokenized_ids_target_ts.ne(tokenizer.mask_token_id)
                current_token_mask = token_mask.clone().repeat(beam,1)
                next_token_mask_list = tokenized_ids_target_ts.ne(tokenizer.mask_token_id).clone().repeat(beam,1)
                current_score = [1.0] * beam
                next_score = []  
                current_sample = tokens.clone().repeat(beam,1)
                for token_index in range(len(tokenized_ids_target)):
                    output_tensors = F.softmax(model(current_sample, padding_mask, index_mask)[0].detach(), dim=-1)
                    for beamsample_index in range(output_tensors.size(0)): 
                        # print(current_score)
                        # print(current_token_mask)
                        # print(next_token_mask_list)
                        # import pdb; pdb.set_trace()
                        current_mask = current_token_mask[beamsample_index].clone()
                        current_sample_ids = current_sample[beamsample_index].clone()
                        current_scores_list = output_tensors[beamsample_index][attention_index:attention_index+len(tokenized_ids_target)]
                        current_scores = torch.gather(current_scores_list,1,tokenized_ids_target_ts.unsqueeze(-1)).to(torch.float64)
                        # import pdb; pdb.set_trace()
                        # current_scores = np.log(current_scores)
                        current_scores = current_scores.squeeze(-1).masked_fill_(~current_mask, 1.0)
                        sorted_current_scores = sorted(current_scores)
                        for score_index in range(len(sorted_current_scores)):
                            if score_index >= beam:
                                break
                            else:
                                if len(next_score) < beam :
                                    next_score.append((current_score[beamsample_index] * sorted_current_scores[score_index]).tolist())
                                    max_score_index = current_scores.tolist().index(sorted_current_scores[score_index])
                                    current_sample_ids_next = current_sample_ids.clone()
                                    current_sample_ids_next[attention_index+max_score_index]=tokenized_ids_target[max_score_index]
                                    if len(current_sample) < len(next_score):
                                        current_sample_list = []
                                        for l in range(current_sample.size(0)):
                                            current_sample_list.append(current_sample[l].unsqueeze_(0))
                                        current_sample_list.append(current_sample_ids_next.unsqueeze_(0))
                                        # import pdb; pdb.set_trace()
                                        current_sample = torch.cat(current_sample_list,dim=0)
                                    else:
                                        current_sample[len(next_score)-1] = current_sample_ids_next
                                        # import pdb;
                                        # pdb.set_trace()
                                    next_token_mask = current_mask.clone()
                                    next_token_mask[max_score_index]= False
                                    next_token_mask_list[len(next_score)-1] = next_token_mask
                                    # print(current_token_mask)
                                    # print(next_token_mask_list)
                                    # import pdb; pdb.set_trace()
                                else:
                                    if token_index!=0:
                                        if (current_score[beamsample_index] * sorted_current_scores[score_index]).tolist() < max(next_score):
                                            replace_score_index = next_score.index(max(next_score))
                                            next_score[replace_score_index]=(current_score[beamsample_index] * sorted_current_scores[score_index]).tolist()
                                            max_score_index = current_scores.tolist().index(sorted_current_scores[score_index])
                                            current_sample_ids_next = current_sample_ids.clone()
                                            current_sample_ids_next[attention_index+max_score_index]=tokenized_ids_target[max_score_index]
                                            current_sample[replace_score_index] = current_sample_ids_next
                                            next_token_mask = current_mask.clone()
                                            next_token_mask[max_score_index]= False
                                            # if len(current_token_mask) < replace_score_index:
                                            #     current_token_mask.append(next_token_mask)
                                            # else:
                                            next_token_mask_list[replace_score_index] = next_token_mask
                                    else:
                                        # import pdb; pdb.set_trace()
                                        if (current_score[beamsample_index] * sorted_current_scores[score_index]).tolist() < max(next_score) and ((current_score[beamsample_index] * sorted_current_scores[score_index]).tolist() not in next_score):
                                            replace_score_index = next_score.index(max(next_score))
                                            next_score[replace_score_index]=(current_score[beamsample_index] * sorted_current_scores[score_index]).tolist()
                                            max_score_index = current_scores.tolist().index(sorted_current_scores[score_index])
                                            current_sample_ids_next = current_sample_ids.clone()
                                            current_sample_ids_next[attention_index+max_score_index]=tokenized_ids_target[max_score_index]
                                            current_sample[replace_score_index] = current_sample_ids_next
                                            next_token_mask = current_mask.clone()
                                            next_token_mask[max_score_index]= False
                                            # if len(current_token_mask) < replace_score_index:
                                            #     current_token_mask.append(next_token_mask)
                                            # else:
                                            next_token_mask_list[replace_score_index] = next_token_mask
                    # import pdb; pdb.set_trace()
                    current_score = next_score
                    next_score = []
                    current_token_mask=next_token_mask_list.clone()   
                # import pdb; pdb.set_trace()
                current_score = np.log(current_score)  
                result.append((max(current_score), False))
                # result.append((np.power(max(current_score), 1/len(tokenized_ids_target)),False)) 
                # result.append((max(current_score), False))
            # import pdb; pdb.set_trace()
            return result

        if self.inftype == "max_beam_based":
            tokenizer = self.tokenizer
            model = self.model
            result = []
            device = self._device
            beam = 1
            # import pdb; pdb.set_trace()
            max_seq_length = self._max_length
            for context, continuation in tqdm([req.args for req in requests]):
                source = context
                target = continuation
                tokenized_ids_source = tokenizer.encode(source, add_special_tokens = False)
                tokenized_ids_all = tokenizer.encode(source+target, add_special_tokens = False)
                tokenized_ids_target = tokenized_ids_all[len(tokenized_ids_source):]
                sample_len = len(tokenized_ids_target) + len(tokenized_ids_source)
                num_tokens = sample_len + 1
                if num_tokens > max_seq_length:
                    trun_index = max_seq_length - len(tokenized_ids_target) - 1
                    tokenized_ids_source = tokenized_ids_source[:trun_index] 
                    num_tokens = max_seq_length
                
                prev_tokenized_ids_target = np.full(len(tokenized_ids_target), tokenizer.mask_token_id).tolist()
                input_ids = [[tokenizer.bos_token_id] + tokenized_ids_source + prev_tokenized_ids_target]
                # assert len(input_ids[0])==max_seq_length
                tokens = torch.tensor(input_ids).to(device)
                
                tokenized_ids_target_ts = torch.tensor(tokenized_ids_target).to(device)
                attention_index = len(tokenized_ids_source) + 1
                token_mask = tokenized_ids_target_ts.ne(tokenizer.mask_token_id)
                current_token_mask = token_mask.clone().repeat(beam,1)
                next_token_mask_list = tokenized_ids_target_ts.ne(tokenizer.mask_token_id).clone().repeat(beam,1)
                current_score = [1.0] * beam
                next_score = []  
                current_sample = tokens.clone().repeat(beam,1)
                for token_index in range(len(tokenized_ids_target)):
                    output_tensors = F.softmax(model(current_sample).logits, dim=-1)
                    for beamsample_index in range(output_tensors.size(0)): 
                        current_mask = current_token_mask[beamsample_index].clone()
                        current_sample_ids = current_sample[beamsample_index].clone()
                        current_scores_list = output_tensors[beamsample_index][attention_index:attention_index+len(tokenized_ids_target)]
                        current_scores = torch.gather(current_scores_list,1,tokenized_ids_target_ts.unsqueeze(-1)).to(torch.float64)
                        current_scores = current_scores.squeeze(-1).masked_fill_(~current_mask, 0.0)
                        sorted_current_scores = sorted(current_scores, reverse=True)
                        for score_index in range(len(sorted_current_scores)):
                            if score_index >= beam:
                                break
                            else:
                                if len(next_score) < beam :
                                    next_score.append((current_score[beamsample_index] * sorted_current_scores[score_index]).tolist())
                                    max_score_index = current_scores.tolist().index(sorted_current_scores[score_index])
                                    current_sample_ids_next = current_sample_ids.clone()
                                    current_sample_ids_next[attention_index+max_score_index]=tokenized_ids_target[max_score_index]
                                    if len(current_sample) < len(next_score):
                                        current_sample_list = []
                                        for l in range(current_sample.size(0)):
                                            current_sample_list.append(current_sample[l].unsqueeze_(0))
                                        current_sample_list.append(current_sample_ids_next.unsqueeze_(0))
                                        current_sample = torch.cat(current_sample_list,dim=0)
                                    else:
                                        current_sample[len(next_score)-1] = current_sample_ids_next
                                    next_token_mask = current_mask.clone()
                                    next_token_mask[max_score_index]= False
                                    next_token_mask_list[len(next_score)-1] = next_token_mask
                                else:
                                    if token_index!=0:
                                        if (current_score[beamsample_index] * sorted_current_scores[score_index]).tolist() > min(next_score):
                                            replace_score_index = next_score.index(min(next_score))
                                            next_score[replace_score_index]=(current_score[beamsample_index] * sorted_current_scores[score_index]).tolist()
                                            max_score_index = current_scores.tolist().index(sorted_current_scores[score_index])
                                            current_sample_ids_next = current_sample_ids.clone()
                                            current_sample_ids_next[attention_index+max_score_index]=tokenized_ids_target[max_score_index]
                                            current_sample[replace_score_index] = current_sample_ids_next
                                            next_token_mask = current_mask.clone()
                                            next_token_mask[max_score_index]= False
                                            next_token_mask_list[replace_score_index] = next_token_mask
                                    else:
                                        if (current_score[beamsample_index] * sorted_current_scores[score_index]).tolist() > min(next_score) and ((current_score[beamsample_index] * sorted_current_scores[score_index]).tolist() not in next_score):
                                            replace_score_index = next_score.index(min(next_score))
                                            next_score[replace_score_index]=(current_score[beamsample_index] * sorted_current_scores[score_index]).tolist()
                                            max_score_index = current_scores.tolist().index(sorted_current_scores[score_index])
                                            current_sample_ids_next = current_sample_ids.clone()
                                            current_sample_ids_next[attention_index+max_score_index]=tokenized_ids_target[max_score_index]
                                            current_sample[replace_score_index] = current_sample_ids_next
                                            next_token_mask = current_mask.clone()
                                            next_token_mask[max_score_index]= False
                                            next_token_mask_list[replace_score_index] = next_token_mask
                    current_score = next_score
                    next_score = []
                    current_token_mask=next_token_mask_list.clone()   

                current_score = np.log(current_score)  
                result.append((max(current_score), False))
            
            return result

        if self.inftype == "max_iteration_base":
            tokenizer = self.tokenizer
            model = self.model
            result = []
            device = self._device
            beam = 1
            max_seq_length = self._max_length
            batch_size = self._batch_size
            max_iter = self._max_iter

            # if self._batch_size == "auto":
            # import pdb; pdb.set_trace()
            
            for i in tqdm(range(0, len(requests), batch_size)):
                batch = requests[i:i + batch_size]
                contexts, continuations = zip(*[(req.args[0], req.args[1]) for req in batch])
                token_list = []
                index_mask_list = []
                attention_index_list = []
                label_list = []
                target_list = []
                for sample_index in range(len(list(contexts))):
                    source = contexts[sample_index]
                    target = continuations[sample_index]
                    tokenized_ids_source = tokenizer.tokenize(source)
                    tokenized_ids_target = tokenizer.tokenize(target)
                    # target_list.append(tokenized_ids_target)
                    sample_len = len(tokenized_ids_source) + len(tokenized_ids_target)
                    num_tokens = sample_len + 1
                    if num_tokens > max_seq_length:
                        trun_index = max_seq_length - len(tokenized_ids_target) - 1
                        tokenized_ids_source = tokenized_ids_source[:trun_index] 
                        num_tokens = max_seq_length
                    prev_tokenized_ids_target = np.full(len(tokenized_ids_target), tokenizer.mask_token_id).tolist()
                    pad_list = [tokenizer.pad for _ in range(max_seq_length-num_tokens)]
                    input_ids = [[tokenizer.bos] + tokenized_ids_source + prev_tokenized_ids_target + pad_list]
                    label_ids = [[tokenizer.bos] + tokenized_ids_source + tokenized_ids_target + pad_list]
                    assert len(input_ids[0])==max_seq_length
                    tokens = torch.tensor(input_ids).to(device)
                    label_tokens = torch.tensor(label_ids).to(device)
                    padding_mask = tokens.ne(tokenizer.pad).type_as(tokens)
                    # attention_index
                    index_mask = np.zeros((max_seq_length, max_seq_length))
                    attention_index = len(tokenized_ids_source) + 1
                    attention_index_list.append(attention_index)
                    attention_index_list.append(attention_index)
                    index_mask[:attention_index, :attention_index]=1
                    index_mask[attention_index:num_tokens, :num_tokens]=1
                    index_mask = torch.tensor(index_mask)
                    index_mask = index_mask.unsqueeze(0).to(device)
                    # import pdb; pdb.set_trace()
                    token_list.append(tokens)
                    index_mask_list.append(index_mask)
                    label_list.append(label_tokens)
                # import pdb; pdb.set_trace()
                batch_tokens = torch.cat(token_list,dim=0)
                batch_index_mask = torch.cat(index_mask_list, dim=0)
                batch_labels = torch.cat(label_list, dim=0)



                if max_iter == 0:
                    max_iter = max(batch_tokens.eq(tokenizer.mask_token_id).sum(1)).tolist()


                decoder_option = {
                    "current_step": 0,
                    "max_step": max_iter,
                    "current_ids": batch_tokens,
                    "current_confidence": None,
                }
                # import pdb; pdb.set_trace()
                for step in range(max_iter):
                    decoder_option["current_step"] = step
                    decoder_option = forward_sample(model=model, tokenizer=tokenizer, decoder_option=decoder_option, labels=batch_labels, padding_mask=padding_mask, index_mask=batch_index_mask)
                # import pdb; pdb.set_trace()
                output_score = torch.sum(decoder_option["current_confidence"], dim=1).tolist()
                for score in output_score:
                    result.append((score,False))
        
            return result

    def generate_until(self, requests):
        import pdb; pdb.set_trace()
        return None 



from transformers import AutoTokenizer, RobertaForMaskedLM
import torch
import json_lines
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import argparse


def main():
    group = argparse.ArgumentParser(description='test')
    group.add_argument('--task_list', type=str, default = "all", help='Either "all" or comma separated list of tasks.')
    group.add_argument('--results_path', type=str, default = "./results.json", help='Path to where the results will be stored.')
    group.add_argument('--num_fewshot', type=int, default = 0, help='Number of few-shot prompts.')
    group.add_argument('--inftype',type=str, default='diff_gram_based', help='')
    group.add_argument('--max-iter',type=int, default=1)
    group.add_argument('--model_path',type=str, default='', help='')
    group.add_argument('--seq_length',type=int, default=512)
    args = group.parse_args()


    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = RobertaForMaskedLM.from_pretrained(args.model_path)
    model = model.to(device)
    model.eval()
    task_list = args.task_list.split(',')
    task_dict = tasks.get_task_dict(task_list)
    adaptor = EvalHarnessAdaptor(args,model, tokenizer,device)
    
    results = evaluator.evaluate(adaptor, task_dict)
    # import pdb; pdb.set_trace()
    print(results["results"])
    print(results["n-shot"])



if __name__ == '__main__':
    main()
