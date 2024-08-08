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
from lm_eval.logging_utils import add_env_info
import torch.nn.functional as F
import math
# from lm_eval.tasks import ALL_TASKS
# from pretrain_gpt import model_provider
import numpy as np
import time
import random
import torch
from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron.core.enums import ModelType
from megatron.core import mpu
from megatron.training import setup_model_and_optimizer, get_model
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region

from megatron.utils import get_ltor_masks_and_position_ids, unwrap_model
from megatron.p2p_communication import recv_forward, send_forward
import pickle
import json
from accelerate import (
    Accelerator,
    DistributedType,
    InitProcessGroupKwargs,
    find_executable_batch_size,
)

from lm_eval.models.utils import (
    Collator,
    clear_torch_cache,
    get_dtype,
    pad_and_concat,
    stop_sequences_criteria,
)

from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.model.distributed import DistributedDataParallel as LocalDDP
from megatron.model.module import Float16Module
import deepspeed
from deepspeed.accelerator import get_accelerator


class EvalHarnessAdaptor(LM):
    def __init__(self, model, tokenizer):
        args = get_args()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.VOCAB_SIZE = tokenizer.vocab_size
        self.mask_token_id = tokenizer.mask
        self.bos_token_id = tokenizer.bos
        self.pad_token_id = tokenizer.pad
        self.eos_token_id = tokenizer.eos
        
    
        # For ds we split into mini batches and then micro batches to keep pipelining api happy.
        # With Megatron we just go to micro_batches directly
        self.inftype = args.inftype
        self._max_length = args.seq_length
        self._max_iter = args.max_iter
        self.ngarm = args.ngram
        self.beam = args.beam
        self.has_attention_masking = args.has_attention_masking

        self.max_batch_size = 64
        self._batch_size = args.micro_batch_size
        self.cache_hook = CacheHook(None)
        self._rank = 0
        self._world_size = 1
        self.is_main = args.rank == 0
        self.is_local_main = args.local_rank == 0
        self._device = get_accelerator().current_device_name()
        self.is_model_parallel = mpu.get_tensor_model_parallel_world_size() > 1
        self.is_pipe_parallel = mpu.get_pipeline_model_parallel_world_size() > 1
        self.is_data_parallel = mpu.get_data_parallel_world_size() > 1
        self.adaptive_seq_len = args.adaptive_seq_len
        if self.is_data_parallel and args.moe_expert_parallel_size == 1: # For MoE model, allow a "fake data parallel" in order to partition model into multiple gpus
            raise NotImplementedError("Data parallelism is currently not supported for evaluation")

        self.is_last_stage = True if not self.is_pipe_parallel else mpu.is_pipeline_last_stage()  # only the last stage of the pipeline model will receive the logits

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
        if self.inftype == "max_beam_based":
            tokenizer = self.tokenizer
            model = self.model
            result = []
            device = self._device
            beam = self.beam
            max_seq_length = self._max_length
            for context, continuation in tqdm([req.args for req in requests]):
                # import pdb; pdb.set_trace()
                source = context
                target = continuation
                tokenized_ids_source = tokenizer.tokenize(source)
                tokenized_ids_all = tokenizer.tokenize(source+target)
                tokenized_ids_target = tokenized_ids_all[len(tokenized_ids_source):]
                sample_len = len(tokenized_ids_target) + len(tokenized_ids_source)
                num_tokens = sample_len + 1
                if num_tokens > max_seq_length:
                    trun_index = max_seq_length - len(tokenized_ids_target) - 1
                    tokenized_ids_source = tokenized_ids_source[len(tokenized_ids_source)-trun_index:] 
                    num_tokens = max_seq_length
                pad_list = [tokenizer.pad for _ in range(max_seq_length-num_tokens)]
                prev_tokenized_ids_target = np.full(len(tokenized_ids_target), tokenizer.mask).tolist()
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
                token_mask = tokenized_ids_target_ts.ne(tokenizer.mask)
                current_token_mask = token_mask.clone().repeat(beam,1)
                next_token_mask_list = tokenized_ids_target_ts.ne(tokenizer.mask).clone().repeat(beam,1)
                current_score = [1.0] * beam
                next_score = []  
                # import pdb; pdb.set_trace()
                current_sample = tokens.clone().repeat(beam,1)
                for token_index in range(len(tokenized_ids_target)):
                    current_padding_mask = current_sample.ne(tokenizer.pad).type_as(current_sample)
                    current_index_mask = index_mask.clone().repeat(beam,1,1)
                    output_tensors = F.softmax(model(current_sample, current_padding_mask, current_index_mask)[0].detach(), dim=-1)
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
                                        if (current_score[beamsample_index] * sorted_current_scores[score_index]).tolist() > min(next_score):
                                            replace_score_index = next_score.index(min(next_score))
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
                    # import pdb; pdb.set_trace()
                    current_score = next_score
                    next_score = []
                    current_token_mask=next_token_mask_list.clone()   

                current_score = np.log(current_score)  
                result.append((max(current_score), False))
            
            return result

        else:
            print('Not Implementation Error')


    def generate_until(self, requests):
        import pdb; pdb.set_trace()
        return None 


import deepspeed
from megatron.initialize import initialize_megatron
import megatron
from megatron.training import get_model
from megatron.arguments import core_transformer_config_from_args
from megatron.model import GeBertModel
from megatron.initialize import set_jit_fusion_options
from megatron.checkpointing import load_checkpoint

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

def tasks_args(parser):
    """Provide extra arguments required for tasks."""
    
    #common setting args
    group = parser.add_argument_group(title='Evaluation options')
    group.add_argument('--task_list', type=str, default = "all", help='Either "all" or comma separated list of tasks.')
    group.add_argument('--num_fewshot', type=int, default = 0, help='Number of few-shot prompts.')
    group.add_argument('--adaptive_seq_len',  default = False, action='store_true',
                       help='Should the sequence length be adapted to the batch during evaluation, if in fp16 the results will be slightly different due to numerical errors but greatly speed up evaluation.')
    group.add_argument('--eval_fp32',  default = False, action='store_true', help='Should the evaluation run in fp32')

    # our inference setting args
    group.add_argument('--has-attention-masking', action='store_true', help='If adopt the attention masking.')
    group.add_argument('--inftype',type=str, default='max_beam_based', help='')
    group.add_argument('--max-iter',type=int, default=1)
    group.add_argument('--ngram',type=int, default=1)
    group.add_argument('--beam',type=int, default=1)
    
    # length_prediction args for generation tasks
    group.add_argument('--length-predict', action='store_true', help='if we adopt length prediction')
    group.add_argument('--max-predict-length', type=int, help='the maximun predicted length')
    group.add_argument('--length-factor', type=float, default=0.1, help='length loss factor')
    group.add_argument('--load-LP-module', action='store_true', help='if we load le`ngth prediction')
    
    return parser


def main():
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
    


    start = time.time()
    device = torch.device("cuda")
    initialize_megatron(extra_args_provider=tasks_args, args_defaults={'tokenizer_type': 'HFTokenizer'})

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

    model.eval()
    task_list = args.task_list.split(',')
    task_dict = tasks.get_task_dict(task_list)
    adaptor = EvalHarnessAdaptor(model, tokenizer)
    # if args.num_fewshot
    if args.num_fewshot !=0:
        for task in task_dict.keys():
            if isinstance(task_dict[task], tuple):
                if task_dict[task][1] is None:
                    continue
                else:
                    task_dict[task][1].set_config(key="num_fewshot", value=5)
            else:
                task_dict[task].set_config(key="num_fewshot", value=5)
    results = evaluator.evaluate(adaptor, task_dict)
    print(results["results"])
    print(results["n-shot"])



if __name__ == '__main__':
    main()
