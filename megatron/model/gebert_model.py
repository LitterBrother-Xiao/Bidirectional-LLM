# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""BERT model...."""

import torch
import torch.nn.functional as F
from megatron import get_args
from megatron.core import tensor_parallel
from megatron.model.enums import AttnMaskType
from megatron.model.language_model import parallel_lm_logits
from megatron.model.language_model import get_language_model
from megatron.model import LayerNorm
from megatron.model.utils import openai_gelu, erf_gelu
from megatron.model.utils import get_linear_layer
from megatron.model.utils import init_method_normal
from megatron.model.utils import scaled_init_method_normal
from .module import MegatronModule
from torch import Tensor, nn

def gebert_extended_attention_mask(attention_mask, index_attention_mask = None):
    # We create a 3D attention mask from a 2D tensor mask.
    # [b, 1, s]
    attention_mask_b1s = attention_mask.unsqueeze(1)
    # [b, s, 1]
    attention_mask_bs1 = attention_mask.unsqueeze(2)
    # [b, s, s]
    attention_mask_bss = attention_mask_b1s * attention_mask_bs1
    # [b, 1, s, s]
    extended_attention_mask = attention_mask_bss.unsqueeze(1)
    if index_attention_mask is not None:
        # extended_attention_mask = extended_attention_mask + index_attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask * index_attention_mask.unsqueeze(1)
    # Convert attention mask to binary:
    extended_attention_mask = (extended_attention_mask < 0.5)
    return extended_attention_mask

def bert_position_ids(token_ids):
    # Create position ids
    seq_length = token_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=token_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(token_ids)

    return position_ids

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

class BertLMHead(MegatronModule):
    """Masked LM head for Bert

    Arguments:
        config: TransformerConfig object
        mpu_vocab_size: model parallel size of vocabulary.
        hidden_size: hidden size
        parallel_output: whether output logits being distributed or not.
    """

    def __init__(self, mpu_vocab_size, hidden_size, config, parallel_output):
        super().__init__(config=config)

        args = get_args()
        self.bias = torch.nn.Parameter(torch.zeros(mpu_vocab_size))
        tensor_parallel.set_tensor_model_parallel_attributes(self.bias, True, 0, 1)
        self.parallel_output = parallel_output

        self.dense = get_linear_layer(hidden_size, hidden_size, config.init_method, gather_params_on_init=args.zero_stage == 3)
        setattr(self.dense.weight, 'sequence_parallel', config.sequence_parallel)
        setattr(self.dense.bias, 'sequence_parallel', config.sequence_parallel)

        self.layernorm = LayerNorm(hidden_size,
                                   eps=config.layernorm_epsilon,
                                   sequence_parallel=config.sequence_parallel)
        self.gelu = torch.nn.functional.gelu
        if args.openai_gelu:
            self.gelu = openai_gelu
        elif args.onnx_safe:
            self.gelu = erf_gelu

    def forward(self, hidden_states, word_embeddings_weight):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        output = parallel_lm_logits(hidden_states,
                                    word_embeddings_weight,
                                    self.parallel_output,
                                    bias=self.bias)
        return output

def post_language_model_processing(lm_output, pooled_output,
                                   lm_head, binary_head,
                                   lm_labels,
                                   logit_weights,
                                   fp16_lm_cross_entropy):
    # Output.
    lm_logits = lm_head(
        lm_output, logit_weights)

    # binary_logits = None
    # if binary_head is not None:
    #     binary_logits = binary_head(pooled_output)
    

    if lm_labels is None:
        # [s b h] => [b s h]
        return lm_logits.transpose(0,1).contiguous(), lm_output.transpose(0,1).contiguous()
    else:
        # [b s] => [s b]
        lm_labels = lm_labels.transpose(0,1).contiguous()
        # lm_logits : [s, b, h] and lm_labels: [s, b]
        if fp16_lm_cross_entropy:
            assert lm_logits.dtype == torch.half
            lm_loss = tensor_parallel.vocab_parallel_cross_entropy(lm_logits, lm_labels)
        else:
            lm_loss = tensor_parallel.vocab_parallel_cross_entropy(lm_logits.float(),
                                                        lm_labels)
        # [s, b] => [b s]
        lm_loss = lm_loss.transpose(0,1).contiguous()
        return lm_loss, lm_output
    
def _mean_pooling(enc_feats, src_masks):
    # enc_feats: T x B x C
    # src_masks: B x T or None
    # import pdb;pdb.set_trace()
    if src_masks is None:
        enc_feats = enc_feats.mean(0)
    else:
        src_masks = (src_masks).transpose(0, 1).type_as(enc_feats)
        enc_feats = (
            (enc_feats / src_masks.sum(0)[None, :, None]) * src_masks[:, :, None]
        ).sum(0)
    return enc_feats



class GeBertModel(MegatronModule):
    """Bert Language model."""

    def __init__(self,
                 config,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True,
                 return_moe_loss=False):
        super().__init__(config=config)
        args = get_args()

        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.return_moe_loss = return_moe_loss
        self.language_model, self._language_model_key = get_language_model(
            config=config,
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.padding,
            pre_process=self.pre_process,
            post_process=self.post_process,
            num_experts=args.num_experts)
        self.initialize_word_embeddings()
        if self.post_process:
            self.lm_head = BertLMHead(
                self.shared_embedding_or_output_weight().size(0),
                config.hidden_size, config, parallel_output)
            self.binary_head = None
            self._lm_head_key = 'lm_head'

        if args.length_predict:
            # import pdb; pdb.set_trace()
            self.length_head = get_linear_layer(config.hidden_size, args.max_predict_length, config.init_method, args.zero_stage == 3)
            self._length_head_key = 'length_head'

    def forward_length(self, enc_feats, src_masks):
        # import pdb;pdb.set_trace()
        enc_feats = _mean_pooling(enc_feats.transpose(0, 1), src_masks.bool())
        length_out = F.linear(enc_feats, self.length_head.weight)
        return F.log_softmax(length_out, -1)
    
    def forward_length_prediction(self, length_out, length_tgt=None):
        if length_tgt is not None:
            # obtain the length target
            length_tgt = length_tgt.long()
            length_tgt = length_tgt.clamp(min=0, max=2048)
        else:
            pred_lengs = length_out.max(-1)[1]
            length_tgt = pred_lengs
        return length_tgt


    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, bert_model_input, attention_mask, index_attention_mask=None,
                tokentype_ids=None, lm_labels=None):
        # index_attention_mask: bsz * seq_len * seq_len \
        # enable the model tp suit the generation task
        # import pdb; pdb.set_trace()
        if index_attention_mask is not None:
            extended_attention_mask = (index_attention_mask.unsqueeze(1) < 0.5)
        else:
            extended_attention_mask = gebert_extended_attention_mask(attention_mask)
        input_ids = bert_model_input
        position_ids = bert_position_ids(input_ids)

        lm_output = self.language_model(
            input_ids,
            position_ids,
            extended_attention_mask,
            tokentype_ids=tokentype_ids
        )
        pooled_output = None
        if self.post_process:
            if self.binary_head is None:
                lm_output, moe_losses = lm_output
            lm_output = post_language_model_processing(lm_output, pooled_output,
                                                       self.lm_head, self.binary_head,
                                                       lm_labels,
                                                       self.shared_embedding_or_output_weight(),
                                                       self.fp16_lm_cross_entropy)
            return *lm_output, moe_losses if self.return_moe_loss else lm_output
        else:
            return lm_output


    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """For easy load when model is combined with other heads,
        add an extra key."""
        args = get_args()

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(prefix=prefix,
                                                                 keep_vars=keep_vars)
        if self.post_process:
            state_dict_[self._lm_head_key] \
                = self.lm_head.state_dict_for_save_checkpoint(prefix=prefix,
                                                              keep_vars=keep_vars)
        if args.length_predict:
            state_dict_[self._length_head_key] \
                = self.length_head.state_dict(prefix=prefix,
                                                              keep_vars=keep_vars)
        # if self.post_process and self.add_binary_head:
        #     state_dict_[self._binary_head_key] \
        #         = self.binary_head.state_dict(prefix=prefix, keep_vars=keep_vars)
        # Save word_embeddings.
        if self.post_process and not self.pre_process:
            state_dict_[self._word_embeddings_for_head_key] \
                = self.word_embeddings.state_dict(prefix=prefix, keep_vars=keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""
        args = get_args()
        self.language_model.load_state_dict(
            state_dict[self._language_model_key], strict=strict)
        if self.post_process:
            self.lm_head.load_state_dict(
                state_dict[self._lm_head_key], strict=strict)
        # if args.length_predict:
        #     self.length_head.load_state_dict(
        #         state_dict[self._length_head_key], strict=strict)
        # if self.post_process and self.add_binary_head:
        #     self.binary_head.load_state_dict(
        #         state_dict[self._binary_head_key], strict=strict)
        # Load word_embeddings.
        if self.post_process and not self.pre_process:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)
        