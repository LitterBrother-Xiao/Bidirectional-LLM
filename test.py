# import torch

# model_dict = []

# megatron_lm_weights = torch.load('/nvme/xys/checkpoint/megatron_ckpt/pretraining_gebert/gebert_pretraining_large/iter_0028000/mp_rank_00/model_optim_rng.pt', map_location=torch.device('cpu'))

# ds_weight = torch.load("/nvme/xys/DeepSpeedExamples-master/training/HelloDeepSpeed/experiment_out/experiment_deepspeed_gebert_12_24/bert_pretrain.2023.12.23.21.51.50.addjtvxg/global_step350000/mp_rank_00_model_states.pt", map_location=torch.device('cpu'))
# ds_weight_dict = ds_weight['module']
# import pdb;pdb.set_trace()

# ds_weight_dict['encoder.embeddings.position_ids'] = torch.arange(0,1044)

# ds_weight_dict['encoder.embeddings.word_embeddings.weight'] = megatron_lm_weights['model']['language_model']['embedding']['word_embeddings']['weight']
# ds_weight_dict['encoder.embeddings.position_embeddings.weight'] = megatron_lm_weights['model']['language_model']['embedding']['position_embeddings']['weight']

# ds_weight_dict['encoder.encoder.layer.0.attention.self.query.weight'] = megatron_lm_weights['model']['language_model']['encoder']['layers.0.self_attention.query_key_value.weight']
# ds_weight_dict['encoder.encoder.layer.0.attention.self.query.bias'] = megatron_lm_weights['model']['language_model']['encoder']['layers.0.self_attention.query_key_value.bias']
# ds_weight_dict[]



# 'encoder.embeddings.LayerNorm.weight', 
# 'encoder.embeddings.LayerNorm.bias', 
# 'encoder.encoder.layer.0.attention.self.query.weight', 
# 'encoder.encoder.layer.0.attention.self.query.bias', 
# 'encoder.encoder.layer.0.attention.self.key.weight', 
# 'encoder.encoder.layer.0.attention.self.key.bias', 
# 'encoder.encoder.layer.0.attention.self.value.weight', 
# 'encoder.encoder.layer.0.attention.self.value.bias', 
# 'encoder.encoder.layer.0.attention.output.dense.weight', 
# 'encoder.encoder.layer.0.attention.output.dense.bias', 
# 'encoder.encoder.layer.0.attention.output.LayerNorm.weight', 
# 'encoder.encoder.layer.0.attention.output.LayerNorm.bias', 
# 'encoder.encoder.layer.0.intermediate.dense.weight', 
# 'encoder.encoder.layer.0.intermediate.dense.bias', 
# 'encoder.encoder.layer.0.output.dense.weight', 
# 'encoder.encoder.layer.0.output.dense.bias', 
# 'encoder.encoder.layer.0.output.LayerNorm.weight', 
# 'encoder.encoder.layer.0.output.LayerNorm.bias',


# 'layers.0.input_layernorm.weight', 
# 'layers.0.input_layernorm.bias', 
# 'layers.0.self_attention.query_key_value.weight', 
# 'layers.0.self_attention.query_key_value.bias', 
# 'layers.0.self_attention.dense.weight', 
# 'layers.0.self_attention.dense.bias', 
# 'layers.0.post_attention_layernorm.weight', 
# 'layers.0.post_attention_layernorm.bias', 
# 'layers.0.mlp.dense_h_to_4h.weight', 
# 'layers.0.mlp.dense_h_to_4h.bias', 
# 'layers.0.mlp.dense_4h_to_h.weight', 
# 'layers.0.mlp.dense_4h_to_h.bias'